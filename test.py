import os
import glob
import time
import cv2
import torch
import numpy as np
from ultralytics import YOLO

# Verifica della disponibilità di GPU e impostazione del dispositivo
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Dispositivo in uso: {device}")

# Caricamento modelli pre-addestrati

# Rilevamento oggetti 2D (YOLOv8)
yolo_model = YOLO("yolov8m.pt")

# Stima della profondita monoculare (MiDaS)
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.to(device)
midas.eval()

# Trasformazioni di input per MiDaS
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform

# Mappatura delle classi di veicoli rilevate da YOLOv8
TARGET_CLASSES = {
    0: "Person",
    1: "Bicycle",
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

# Stato globale per il tracciamento degli oggetti tra i frame
tracked_objects = []
SMOOTHING_ALPHA = 0.25  # Peso del nuovo valore: 0.25 nuovo + 0.75 storico

# Parametri per la conversione della profondità stimata in metri
DEPTH_SCALE = 3500.0
EPSILON = 1e-4

def extract_roi_depth(depth_map, x1, y1, x2, y2, crop_ratio=0.5):
    # Estrae la profondità da una regione di interesse (ROI) centrata all'interno della bounding box
    box_w = x2 - x1
    box_h = y2 - y1
    
    # Calcolo dei limiti della ROI centrata
    cx1 = int(x1 + box_w * (1 - crop_ratio) / 2)
    cx2 = int(x2 - box_w * (1 - crop_ratio) / 2)
    cy1 = int(y1 + box_h * (1 - crop_ratio) / 2)
    cy2 = int(y2 - box_h * (1 - crop_ratio) / 2)
    
    roi = depth_map[cy1:cy2, cx1:cx2]
    if roi.size == 0:
        return 0.0
    
    # Calcolo della mediana per ridurre l'influenza di outlier
    return float(np.median(roi))

def apply_temporal_smoothing(current_center, raw_distance, max_pixel_drift=60):
    # Associa il veicolo rilevato nel frame corrente con quelli tracciati nei frame precedenti e applica un filtro passa-basso (EMA) per stabilizzare la stima della distanza
    global tracked_objects
    best_match_idx = -1
    min_dist = float("inf")
    
    # Cerca il veicolo più vicino tra quelli tracciati
    for idx, obj in enumerate(tracked_objects):
        dist = np.linalg.norm(np.array(current_center) - np.array(obj["center"]))
        if dist < min_dist and dist < max_pixel_drift:
            min_dist = dist
            best_match_idx = idx
            
    if best_match_idx != -1:
        # Aggiornamento con filtro passa-basso (EMA)
        smoothed_dist = (SMOOTHING_ALPHA * raw_distance) + ((1 - SMOOTHING_ALPHA) * tracked_objects[best_match_idx]["distance"])
        return smoothed_dist, best_match_idx
    else:
        # Nuovo oggetto rilevato
        return raw_distance, None

def process_single_frame(frame, width, height):
    # Elabora il frame corrente, genera la mappa termica e restituisce la vista combinata
    global tracked_objects

    # --- MiDaS ---
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_batch = transform(img_rgb).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=(height, width),
            mode="bicubic",
            align_corners=False
        ).squeeze()
        depth_map = prediction.cpu().numpy()

    # Generazione mappa termica a colori per la vista affiancata
    depth_norm = cv2.normalize(depth_map, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    depth_heatmap = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)

    # --- YOLOv8 ---
    results = yolo_model(frame, verbose=False)[0]
    
    current_frame_tracked = []
    detections_log = []

    # --- Integrazione Bounding Box, ROI Depth e Smoothing ---
    for box in results.boxes:
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())

        if cls_id in TARGET_CLASSES and conf > 0.40:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            label_text = TARGET_CLASSES[cls_id]

            # Estrazione valore depth relativo sul veicolo
            raw_depth_val = extract_roi_depth(depth_map, x1, y1, x2, y2, crop_ratio=0.5)
            
            # Conversione in metri stimati
            est_distance = DEPTH_SCALE / (raw_depth_val + EPSILON)
            
            # Filtraggio per distanze plausibili
            if 1.0 <= est_distance <= 80.0:
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                
                # Applicazione del filtro passa-basso per stabilizzare la stima della distanza
                smoothed_distance, match_idx = apply_temporal_smoothing(center, est_distance)
                current_frame_tracked.append({"center": center, "distance": smoothed_distance})

                # Definizione soglia di allerta in base alla classe del veicolo
                soglia_allerta = 15.0 if cls_id in (0, 1) else 10.0
                
                if smoothed_distance < soglia_allerta:
                    box_color = (0, 0, 255)  # Rosso -> pericolo / frenata
                elif cls_id in (0, 1):
                    box_color = (0, 255, 255)  # Giallo -> pedone/ciclista a distanza di sicurezza
                else:
                    box_color = (0, 255, 0)  # Verde -> veicolo a distanza di sicurezza

                # Salvataggio dati per il log a terminale
                detections_log.append({
                    "classe": label_text,
                    "distanza": smoothed_distance,
                    "conf": conf,
                    "box": (x1, y1, x2 - x1, y2 - y1)
                })

                # --- Output grafico ---
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                
                # Disegno di un rettangolo più piccolo all'interno della bounding box per indicare la ROI di profondità
                bw, bh = x2 - x1, y2 - y1
                rx1, rx2 = int(x1 + bw * 0.25), int(x2 - bw * 0.25)
                ry1, ry2 = int(y1 + bh * 0.25), int(y2 - bh * 0.25)
                cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (255, 255, 0), 1)

                # Etichetta con classe e distanza in metri
                tag = f"{label_text}: {smoothed_distance:.1f}m"
                cv2.putText(frame, tag, (x1, max(y1 - 8, 15)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

    # Aggiornamento stato oggetti attivi per il fotogramma successivo
    tracked_objects = current_frame_tracked

    # Intestazioni grafiche per i due riquadri
    cv2.putText(frame, "RGB Frame + Detection", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(depth_heatmap, "Depth Map (MiDaS)", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Vista a doppio schermo affiancata
    combined_view = np.hstack((frame, depth_heatmap))
    
    return combined_view, detections_log

def run_perception_pipeline(source_type="video", source_path="data/test_video.mp4", save_output=True, output_filename="output_dimostrazione.mp4"):
    # Gestione dell'input per video o cartella di frame fotografici
    writer = None

    if source_type == "video":
        cap = cv2.VideoCapture(source_path)
        if not cap.isOpened():
            print(f"Errore: Impossibile aprire la sorgente video: {source_path}")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Sorgente Video: {width}x{height} a {fps:.1f} FPS ({total_frames} frame totali)")

        # Inizializzazione salvataggio video doppio schermo
        if save_output:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_filename, fourcc, fps if fps > 0 else 25.0, (width * 2, height))

        frame_idx = 0
        while cap.isOpened():
            t_start = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            combined_view, detections = process_single_frame(frame, width, height)
            current_fps = 1.0 / (time.time() - t_start)

            # Output di dettaglio nel terminale
            print(f"[{frame_idx:04d}/{total_frames}] FPS: {current_fps:.1f} | Veicoli rilevati: {len(detections)}")
            for d in detections:
                print(f"   -> {d['classe']}: {d['distanza']:.1f}m | conf={d['conf']:.2f} | box [x={d['box'][0]}, y={d['box'][1]}, w={d['box'][2]}, h={d['box'][3]}]")

            if writer:
                writer.write(combined_view)

            # Visualizzazione a schermo
            cv2.imshow("3D Perception Pipeline - Monocular Vision", combined_view)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()

    elif source_type == "foto":
        # Ricerca ordinata delle immagini nella directory fornita
        valid_exts = ("*.png", "*.jpg", "*.jpeg")
        image_files = []
        for ext in valid_exts:
            image_files.extend(glob.glob(os.path.join(source_path, ext)))
        image_files = sorted(image_files)

        if not image_files:
            print(f"Errore: Nessun fotogramma trovato nel percorso: {source_path}")
            return

        first_img = cv2.imread(image_files[0])
        height, width = first_img.shape[:2]
        total_frames = len(image_files)
        print(f"Sorgente Cartella: {total_frames} fotogrammi trovati in {source_path}")

        # Inizializzazione salvataggio per sequenza immagini (default 10 FPS)
        if save_output:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_filename, fourcc, 10.0, (width * 2, height))

        for idx, img_path in enumerate(image_files):
            t_start = time.time()
            frame = cv2.imread(img_path)
            if frame is None:
                continue

            combined_view, detections = process_single_frame(frame, width, height)
            current_fps = 1.0 / (time.time() - t_start)

            # Output di dettaglio nel terminale
            print(f"[Frame {idx+1:04d}/{total_frames}] {os.path.basename(img_path)} | FPS: {current_fps:.1f} | Veicoli rilevati: {len(detections)}")
            for d in detections:
                print(f"   -> {d['classe']}: {d['distanza']:.1f}m | conf={d['conf']:.2f} | box [x={d['box'][0]}, y={d['box'][1]}, w={d['box'][2]}, h={d['box'][3]}]")

            if writer:
                writer.write(combined_view)

            # Visualizzazione a schermo
            cv2.imshow("3D Perception Pipeline - Monocular Vision", combined_view)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    if writer:
        writer.release()
        print(f"\nVideo dimostrativo salvato con successo: {output_filename}")

    cv2.destroyAllWindows()

def main():
    # Modalità di esecuzione -> "video" per un video, "foto" per una cartella di immagini
    modalita = "video"
    
    if modalita == "video":
        run_perception_pipeline(
            source_type="video", 
            source_path="data/Video test/test_video1.mp4", 
            save_output=True, 
            output_filename="output_dimostrazione.mp4"
        )
    elif modalita == "foto":
        run_perception_pipeline(
            source_type="foto", 
            source_path="data/Foto test/kitti", 
            save_output=True, 
            output_filename="output_dimostrazione.mp4"
        )

if __name__ == "__main__":
    main()