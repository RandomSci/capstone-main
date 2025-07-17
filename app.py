import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tempfile
import os
import json
from typing import List, Tuple, Dict
import logging
from dataclasses import dataclass
from PIL import Image
import time
import warnings
import threading
import gc
import psutil

os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'
os.environ['STREAMLIT_SERVER_ENABLE_STATIC_SERVING'] = 'false'
os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.basicConfig(level=logging.ERROR)
log = logging.getLogger(__name__)

mdl_cache = {}
mdl_lock = threading.Lock()

YOLO_MODELS = {
    'YOLOv8n (Fastest)': 'yolov8n.pt',
    'YOLOv8s (Balanced)': 'yolov8s.pt',
    'YOLOv8m (Better Accuracy)': 'yolov8m.pt',
    'YOLOv8l (High Accuracy)': 'yolov8l.pt',
}

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def cleanup_memory():
    gc.collect()
    if len(mdl_cache) > 2:
        oldest_key = next(iter(mdl_cache))
        del mdl_cache[oldest_key]
        gc.collect()

@st.cache_resource
def get_mdl(mdl_name: str = 'YOLOv8s (Balanced)'):
    global mdl_cache
    
    memory_mb = get_memory_usage()
    if memory_mb > 6000:
        st.warning(f"⚠️ High memory usage: {memory_mb:.0f}MB. Cleaning up...")
        cleanup_memory()
    
    with mdl_lock:
        mdl_path = YOLO_MODELS.get(mdl_name, 'yolov8s.pt')
        
        if mdl_name not in mdl_cache:
            try:
                from ultralytics import YOLO
                
                st.info(f"🚂 Loading {mdl_name} on Railway...")
                
                if len(mdl_cache) >= 2:
                    cleanup_memory()
                
                mdl_cache[mdl_name] = YOLO(mdl_path)
                
                dummy = np.zeros((640, 640, 3), dtype=np.uint8)
                mdl_cache[mdl_name].predict(dummy, verbose=False)
                
                memory_after = get_memory_usage()
                st.success(f"✅ {mdl_name} loaded! Memory: {memory_after:.0f}MB")
                
            except ImportError:
                st.error("❌ ultralytics not found. Installing...")
                return None
            except Exception as e:
                st.error(f"❌ Error loading {mdl_name}: {str(e)}")
                return None
                    
    return mdl_cache.get(mdl_name)

st.set_page_config(
    page_title="🚗 Smart Parking Detection",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .railway-info {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #6c757d;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@dataclass
class PSpace:
    id: int
    poly: List[Tuple[float, float]]
    lbl_pos: Tuple[int, int]
    
    def __post_init__(self):
        self.poly_arr = np.array(self.poly, dtype=np.int32)
    
    def scale_to_size(self, original_size: Tuple[int, int], target_size: Tuple[int, int]):
        orig_w, orig_h = original_size
        target_w, target_h = target_size
        
        scale_x = target_w / orig_w
        scale_y = target_h / orig_h
        
        scaled_poly = []
        for x, y in self.poly:
            scaled_poly.append((x * scale_x, y * scale_y))
        
        scaled_lbl_pos = (int(self.lbl_pos[0] * scale_x), int(self.lbl_pos[1] * scale_y))
        
        return PSpace(
            id=self.id,
            poly=scaled_poly,
            lbl_pos=scaled_lbl_pos
        )

class PDetector:
    def __init__(self, frm_sz: Tuple[int, int] = (1280, 720)):
        self.frm_sz = frm_sz
        self.spaces = []
        self.original_spaces = []
        self.mdl_name = 'YOLOv8s (Balanced)'
        
        self.conf_th = 0.3
        self.iou_th = 0.5
        
        self.det_sz_img = (1280, 1280)
        self.det_sz_vid = (800, 800)
        self.skip_frm = 2 
        self.last_det = []
        
    def set_model(self, mdl_name: str):
        self.mdl_name = mdl_name
        
    def load_spaces(self, sp_data: List[Dict]) -> None:
        self.original_spaces = []
        for sp in sp_data:
            space = PSpace(
                id=sp['id'],
                poly=sp['polygon'],
                lbl_pos=tuple(sp['label_position'])
            )
            self.original_spaces.append(space)
    
    def scale_spaces_to_frame(self, frame_shape: Tuple[int, int]):
        if not self.original_spaces:
            return
        
        original_size = (1280, 720)
        current_size = (frame_shape[1], frame_shape[0])
        
        self.spaces = []
        for space in self.original_spaces:
            scaled_space = space.scale_to_size(original_size, current_size)
            self.spaces.append(scaled_space)
    
    def det_veh_simple(self, frm: np.ndarray, model=None) -> List[Tuple[int, int, int, int, str, float]]:
        if model is None:
            m = get_mdl(self.mdl_name)
            if m is None:
                return []
        else:
            m = model
        
        try:
            detection_frame = frm.copy()
            h, w = detection_frame.shape[:2]
            scale_factor = 1.0
            
            if max(h, w) > 1920:
                scale = 1920 / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                detection_frame = cv2.resize(detection_frame, (new_w, new_h))
                scale_factor = 1 / scale
            
            results = m.predict(detection_frame, verbose=False, conf=self.conf_th)
            
            if not results or len(results[0].boxes) == 0:
                return []
            
            dets = []
            boxes = results[0].boxes
            
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls_id = int(box.cls[0].cpu().numpy())
                cls_nm = m.names[cls_id]
                
                x1 *= scale_factor
                y1 *= scale_factor
                x2 *= scale_factor
                y2 *= scale_factor
                
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                if cls_nm.lower() in ['car', 'truck', 'bus', 'motorcycle']:
                    dets.append((x1, y1, x2, y2, cls_nm, float(conf)))
            
            return dets
            
        except Exception as e:
            log.error(f"Detection error: {str(e)}")
            return []
    
    def chk_occ_simple(self, dets: List[Tuple[int, int, int, int, str, float]]) -> Dict[int, Dict]:
        occ = {}
        
        for sp in self.spaces:
            occupied = False
            veh = None
            
            for x1, y1, x2, y2, cls_nm, conf in dets:
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                
                result = cv2.pointPolygonTest(sp.poly_arr, (cx, cy), False)
                if result >= 0:
                    occupied = True
                    veh = {
                        'bbox': (x1, y1, x2, y2),
                        'class': cls_nm,
                        'confidence': conf,
                        'center': (cx, cy),
                        'overlap_ratio': 1.0
                    }
                    break
            
            occ[sp.id] = {'occupied': occupied, 'vehicle': veh}
        
        return occ
    
    def draw_res_simple(self, frm: np.ndarray, occ: Dict[int, Dict]) -> np.ndarray:
        res_frm = frm.copy()
        
        for sp in self.spaces:
            sp_inf = occ[sp.id]
            
            if sp_inf['occupied']:
                cv2.polylines(res_frm, [sp.poly_arr], True, (0, 0, 255), 2)
                cv2.putText(res_frm, str(sp.id), sp.lbl_pos, 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                if sp_inf['vehicle']:
                    veh = sp_inf['vehicle']
                    x1, y1, x2, y2 = veh['bbox']
                    cx, cy = veh['center']
                    
                    cv2.rectangle(res_frm, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(res_frm, (cx, cy), 3, (0, 0, 255), -1)
                    cv2.putText(res_frm, f"{veh['class']} {veh['confidence']:.2f}", 
                               (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            else:
                cv2.polylines(res_frm, [sp.poly_arr], True, (0, 255, 0), 2)
                cv2.putText(res_frm, str(sp.id), sp.lbl_pos, 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        occ_cnt = sum(1 for inf in occ.values() if inf['occupied'])
        tot_sp = len(self.spaces)
        avl_sp = tot_sp - occ_cnt
        
        summary = f"Available: {avl_sp}/{tot_sp} | Occupied: {occ_cnt}/{tot_sp}"
        cv2.rectangle(res_frm, (10, 10), (400, 40), (0, 0, 0), -1)
        cv2.putText(res_frm, summary, (15, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return res_frm
    
    def proc_img(self, frm: np.ndarray) -> Tuple[np.ndarray, Dict[int, Dict]]:
        if not self.original_spaces:
            return frm, {}
        
        original_frame = frm.copy()
        self.scale_spaces_to_frame(original_frame.shape)
        
        dets = self.det_veh_simple(original_frame)
        occ = self.chk_occ_simple(dets)
        res_frm = self.draw_res_simple(original_frame, occ)
        
        return res_frm, occ
    
    def proc_vid_frm(self, frm: np.ndarray, frm_num: int = 0, model=None) -> Tuple[np.ndarray, Dict[int, Dict]]:
        if not self.original_spaces:
            return frm, {}
        
        self.scale_spaces_to_frame(frm.shape)
        dets = self.det_veh_simple(frm, model)
        occ = self.chk_occ_simple(dets)
        res_frm = self.draw_res_simple(frm, occ)
        
        return res_frm, occ

def proc_vid_railway(vid_path: str, det: PDetector, prog_bar, stat_txt) -> tuple:
    cap = cv2.VideoCapture(vid_path)
    
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    tot_frms = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    max_frames = tot_frms
    w, h = orig_w, orig_h
    
    out_path = vid_path.replace('.mp4', '_processed.mp4')
    
    model = get_mdl(det.mdl_name)
    if model is None:
        raise RuntimeError("Failed to load YOLO model")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    
    frm_cnt = 0
    processed_cnt = 0
    occ_data = []
    
    while frm_cnt < max_frames:
        ret, frm = cap.read()
        if not ret:
            break
        
        if frm_cnt % det.skip_frm == 0:
            res_frm, occ = det.proc_vid_frm(frm, frm_cnt, model)
            processed_cnt += 1
            
            if occ and processed_cnt % 10 == 0:
                occ_cnt = sum(1 for inf in occ.values() if inf['occupied'])
                occ_data.append({
                    'frame': frm_cnt,
                    'timestamp': frm_cnt / fps,
                    'occupied': occ_cnt,
                    'available': len(occ) - occ_cnt
                })
        else:
            res_frm = frm
        
        out.write(res_frm)
        frm_cnt += 1
        
        if frm_cnt % 30 == 0:
            prog = frm_cnt / max_frames
            prog_bar.progress(min(prog, 1.0))
            stat_txt.text(f"Processing frame {frm_cnt}/{max_frames} ({prog:.1%})")
    
    cap.release()
    out.release()
    
    return out_path, occ_data

class Editor:
    def __init__(self):
        self.def_cfg = {
            "spaces": [
                {"id": 1, "polygon": [[306.061, 493.939], [159.697, 557.879], [226.061, 620.606], [380.606, 537.576]], "label_position": [385, 521]},
                {"id": 2, "polygon": [[459.091, 498.485], [307.576, 581.212], [382.727, 629.091], [538.182, 533.636]], "label_position": [540, 515]},
                {"id": 3, "polygon": [[606.364, 492.121], [467.273, 583.03], [569.697, 628.788], [695.455, 523.636]], "label_position": [698, 508]},
                {"id": 4, "polygon": [[843.939, 512.424], [748.485, 485.152], [637.273, 574.545], [757.273, 625.455]], "label_position": [845, 498]},
                {"id": 5, "polygon": [[961.818, 500.303], [877.273, 477.576], [812.727, 561.212], [923.333, 597.576]], "label_position": [961, 484]},
                {"id": 6, "polygon": [[978.788, 470], [948.788, 544.242], [1038.788, 568.182], [1053.636, 488.182]], "label_position": [1048, 473]},
                {"id": 7, "polygon": [[1124.848, 548.788], [1126.061, 476.667], [1061.515, 460], [1048.485, 528.788]], "label_position": [1126, 462]},
                {"id": 8, "polygon": [[441.515, 437.576], [385.152, 418.182], [256.97, 458.182], [307.879, 490]], "label_position": [252, 470]},
                {"id": 9, "polygon": [[457.879, 495.455], [568.788, 437.576], [503.636, 415.152], [390.0, 460.909]], "label_position": [386, 474]},
                {"id": 10, "polygon": [[698.485, 433.03], [620.606, 415.758], [527.879, 456.97], [608.485, 486.97]], "label_position": [531, 467]},
                {"id": 11, "polygon": [[816.061, 430.303], [736.667, 412.424], [663.939, 456.364], [745.152, 480.303]], "label_position": [660, 468]},
                {"id": 12, "polygon": [[914.545, 427.273], [846.667, 410.606], [789.091, 450.909], [874.545, 473.03]], "label_position": [793, 460]},
                {"id": 13, "polygon": [[995.152, 423.636], [933.03, 408.485], [900.303, 445.152], [976.061, 465.758]], "label_position": [902, 454]},
                {"id": 14, "polygon": [[1063.333, 419.091], [1005.152, 406.061], [989.697, 437.879], [1058.182, 455.758]], "label_position": [993, 449]}
            ]
        }
    
    def render_ed(self, img_arr: np.ndarray = None) -> List[Dict]:
        st.subheader("🛠️ Parking Space Configuration")
        
        with st.expander("📖 Instructions", expanded=False):
            st.markdown("""
            **Configure parking spaces:**
            
            1. Edit JSON configuration below
            2. Define polygon coordinates for each space
            3. Set label positions
            4. Preview with reference image
            """)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.write("**📝 Configuration**")
            
            if 'park_cfg' not in st.session_state:
                st.session_state.park_cfg = json.dumps(self.def_cfg, indent=2)
            
            cfg_txt = st.text_area(
                "Edit parking configuration:",
                value=st.session_state.park_cfg,
                height=400
            )
            
            try:
                cfg_data = json.loads(cfg_txt)
                
                if 'spaces' not in cfg_data:
                    raise ValueError("Must have 'spaces' key")
                
                sps = cfg_data['spaces']
                if not isinstance(sps, list):
                    raise ValueError("'spaces' must be a list")
                
                for i, sp in enumerate(sps):
                    req_keys = ['id', 'polygon', 'label_position']
                    for key in req_keys:
                        if key not in sp:
                            raise ValueError(f"Space {i} missing: {key}")
                    
                    if len(sp['polygon']) < 3:
                        raise ValueError(f"Space {sp['id']} needs 3+ points")
                
                st.success(f"✅ Valid: {len(sps)} spaces")
                st.session_state.park_cfg = cfg_txt
                
                col_a, col_b = st.columns(2)
                
                with col_a:
                    if st.button("🔄 Reset"):
                        st.session_state.park_cfg = json.dumps(self.def_cfg, indent=2)
                        st.rerun()
                
                with col_b:
                    st.download_button(
                        "📥 Download",
                        data=cfg_txt,
                        file_name="parking_config.json",
                        mime="application/json"
                    )
                
                return sps
                
            except json.JSONDecodeError as e:
                st.error(f"❌ JSON Error: {str(e)}")
                return []
            except ValueError as e:
                st.error(f"❌ Error: {str(e)}")
                return []
        
        with col2:
            st.write("**👁️ Preview**")
            
            if img_arr is not None:
                prev_img = img_arr.copy()
                
                try:
                    cfg_data = json.loads(cfg_txt)
                    sps = cfg_data.get('spaces', [])
                    
                    for sp in sps:
                        poly = np.array(sp['polygon'], dtype=np.int32)
                        cv2.polylines(prev_img, [poly], True, (0, 255, 0), 2)
                        
                        lbl_pos = tuple(sp['label_position'])
                        cv2.putText(prev_img, f"#{sp['id']}", lbl_pos,
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    prev_rgb = cv2.cvtColor(prev_img, cv2.COLOR_BGR2RGB)
                    st.image(prev_rgb, use_container_width=True)
                    
                except:
                    st.error("Fix errors for preview")
            else:
                st.info("Upload reference image")

def main():
    st.markdown('<h1 class="main-header">🚗 Smart Parking Detection System</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="railway-info">
        <p>Optimized for better CPU performance.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if 'det' not in st.session_state:
        st.session_state.det = PDetector()
    
    if 'ed' not in st.session_state:
        st.session_state.ed = Editor()
    
    with st.sidebar:
        st.header("⚙️ Settings")
        
        memory_mb = get_memory_usage()
        st.metric("Memory", f"{memory_mb:.0f} MB")
        
        st.write("**🧠 AI Model**")
        
        mdl_choice = st.selectbox(
            "YOLO Model:",
            list(YOLO_MODELS.keys()),
            index=1,  
            help="All models work well on Railway"
        )
        
        if mdl_choice != st.session_state.det.mdl_name:
            st.session_state.det.set_model(mdl_choice)
            st.info(f"Switched to {mdl_choice}")
        
        st.write("**🎛️ Detection**")
        
        conf = st.slider("Confidence", 0.1, 0.9, 0.3, 0.05)
        st.session_state.det.conf_th = conf
        
        iou_th = st.slider("IoU Threshold", 0.1, 0.9, 0.5, 0.05)
        st.session_state.det.iou_th = iou_th
        
        st.write("**🤖 Management**")
        
        if st.button("🔄 Reload Model"):
            if mdl_choice in mdl_cache:
                del mdl_cache[mdl_choice]
            get_mdl(mdl_choice)
        
        if st.button("🗑️ Clear Cache"):
            mdl_cache.clear()
            cleanup_memory()
            st.success("Cache cleared!")

    tab1, tab2, tab3, tab4 = st.tabs(["🔧 Setup", "📷 Images", "🎬 Videos", "📊 Analytics"])
    
    with tab1:
        st.header("🔧 Configuration")
        
        ref_img = st.file_uploader("Upload reference image", type=['jpg', 'jpeg', 'png'])
        
        img_arr = None
        if ref_img:
            img = Image.open(ref_img)
            img_arr = np.array(img)
            if len(img_arr.shape) == 3:
                img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
        
        sps = st.session_state.ed.render_ed(img_arr)
        
        if sps:
            st.session_state.det.load_spaces(sps)
            st.success(f"✅ Loaded {len(sps)} spaces")
    
    with tab2:
        st.header("📷 Image Detection")
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"Model: **{st.session_state.det.mdl_name}**")
        with col2:
            st.info(f"Memory: **{get_memory_usage():.0f}MB**")
        
        up_img = st.file_uploader("Upload image", type=['jpg', 'jpeg', 'png'], key="img_up")
        
        if up_img:
            img = Image.open(up_img)
            img_arr = np.array(img)
            if len(img_arr.shape) == 3:
                img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original")
                st.image(img, use_container_width=True)
            
            with col2:
                st.subheader("Detection Results")
                
                if st.session_state.det.original_spaces:
                    with st.spinner("Processing..."):
                        start_time = time.time()
                        res_frm, occ = st.session_state.det.proc_img(img_arr)
                        proc_time = time.time() - start_time
                        
                        res_rgb = cv2.cvtColor(res_frm, cv2.COLOR_BGR2RGB)
                        st.image(res_rgb, use_container_width=True)
                        
                        st.caption(f"⏱️ {proc_time:.2f}s")
                        
                        if occ:
                            occ_cnt = sum(1 for inf in occ.values() if inf['occupied'])
                            tot = len(occ)
                            avl = tot - occ_cnt
                            occ_rate = (occ_cnt / tot * 100) if tot > 0 else 0
                            
                            col_a, col_b, col_c, col_d = st.columns(4)
                            with col_a:
                                st.metric("Total", tot)
                            with col_b:
                                st.metric("Available", avl)
                            with col_c:
                                st.metric("Occupied", occ_cnt)
                            with col_d:
                                st.metric("Rate", f"{occ_rate:.1f}%")
                            
                            st.subheader("Details")
                            for sp_id, inf in occ.items():
                                if inf['occupied'] and inf['vehicle']:
                                    veh = inf['vehicle']
                                    st.write(f"🔴 **Space {sp_id}**: {veh['class'].title()} ({veh['confidence']:.2f})")
                                else:
                                    st.write(f"🟢 **Space {sp_id}**: Available")
                else:
                    st.warning("⚠️ Configure spaces first!")
    
    with tab3:
        st.header("🎬 Video Processing")
        
        st.info("🚂 Full video processing available on Railway!")
        
        col1, col2 = st.columns(2)
        with col1:
            skip_frms = st.slider("Frame Skip", 1, 10, 2)
            st.session_state.det.skip_frm = skip_frms
        
        with col2:
            max_size = st.selectbox("Max Video Size (MB)", [50, 100, 200, 500], index=2)
        
        up_vid = st.file_uploader("Upload video", type=['mp4', 'avi', 'mov'])
        
        if up_vid and st.session_state.det.original_spaces:
            file_size = len(up_vid.getvalue()) / (1024*1024)
            
            if file_size > max_size:
                st.error(f"Video too large: {file_size:.1f}MB > {max_size}MB")
                return
            
            if st.button("🚀 Process Video", type="primary"):
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_f:
                    tmp_f.write(up_vid.read())
                    vid_path = tmp_f.name
                
                try:
                    start_time = time.time()
                    
                    prog_bar = st.progress(0)
                    stat_txt = st.empty()
                    
                    out_path, occ_data = proc_vid_railway(
                        vid_path, st.session_state.det, prog_bar, stat_txt
                    )
                    
                    proc_time = time.time() - start_time
                    st.success(f"✅ Processed in {proc_time:.1f}s!")
                    
                    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Processed Video")
                            st.video(out_path)
                            
                            with open(out_path, 'rb') as f:
                                st.download_button(
                                    "📥 Download",
                                    data=f.read(),
                                    file_name="processed_video.mp4",
                                    mime="video/mp4"
                                )
                        
                        with col2:
                            st.subheader("Analysis")
                            if occ_data:
                                df = pd.DataFrame(occ_data)
                                st.line_chart(df.set_index('timestamp')[['occupied', 'available']])
                                
                                st.metric("Avg Occupied", f"{df['occupied'].mean():.1f}")
                                st.metric("Peak", f"{df['occupied'].max()}")
                                st.metric("Min", f"{df['occupied'].min()}")
                        
                        os.unlink(out_path)
                    else:
                        st.error("Failed to process video")
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                finally:
                    if os.path.exists(vid_path):
                        os.unlink(vid_path)
        
        elif up_vid:
            st.warning("⚠️ Configure spaces first!")
    
    with tab4:
        st.header("📊 Analytics")
        
        if st.session_state.det.original_spaces:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Performance")
                
                perf_data = {
                    'Model': ['YOLOv8n', 'YOLOv8s', 'YOLOv8m', 'YOLOv8l'],
                    'Speed': ['Fastest', 'Fast', 'Medium', 'Slow'],
                    'Accuracy': ['Good', 'Better', 'Best', 'Excellent'],
                    'Railway': ['✅', '✅', '✅', '⚠️']
                }
                
                df = pd.DataFrame(perf_data)
                st.dataframe(df, use_container_width=True)
            
            with col2:
                st.subheader("System Info")
                
                info = {
                    "Platform": "Railway",
                    "Model": st.session_state.det.mdl_name,
                    "Confidence": st.session_state.det.conf_th,
                    "Memory": f"{get_memory_usage():.0f}MB",
                    "Video Processing": "Full Quality"
                }
                
                st.json(info)
            
            st.subheader("Sample Data")
            
            hours = list(range(24))
            occupancy = np.random.uniform(0.2, 0.9, 24)
            
            df_hourly = pd.DataFrame({
                'Hour': hours,
                'Occupancy': occupancy
            })
            
            st.line_chart(df_hourly.set_index('Hour'))
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Peak", f"{occupancy.max():.1%}")
            with col_b:
                st.metric("Average", f"{occupancy.mean():.1%}")
            with col_c:
                st.metric("Min", f"{occupancy.min():.1%}")
                
        else:
            st.warning("Configure spaces first!")
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p><strong>🚗 Smart Parking Detection System</strong></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()