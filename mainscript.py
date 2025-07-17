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

if "SPACE_ID" in os.environ:
    st.info("🤗 Running on Hugging Face Spaces!")
    os.environ['HF_HOME'] = '/tmp/.huggingface'
    os.environ['TORCH_HOME'] = '/tmp/.torch'
    os.environ['ULTRALYTICS_HOME'] = '/tmp/.ultralytics'

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
}

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def cleanup_memory():
    gc.collect()
    if len(mdl_cache) > 1:
        oldest_key = next(iter(mdl_cache))
        del mdl_cache[oldest_key]
        gc.collect()

@st.cache_resource
def get_mdl(mdl_name: str = 'YOLOv8n (Fastest)'):
    global mdl_cache
    
    memory_mb = get_memory_usage()
    if memory_mb > 12000:
        st.warning(f"⚠️ High memory usage: {memory_mb:.0f}MB. Cleaning up...")
        cleanup_memory()
    
    with mdl_lock:
        mdl_path = YOLO_MODELS.get(mdl_name, 'yolov8n.pt')
        
        if mdl_name not in mdl_cache:
            try:
                from ultralytics import YOLO
                
                if "SPACE_ID" in os.environ:
                    st.info(f"🤗 Loading {mdl_name} on Hugging Face Spaces...")
                else:
                    st.info(f"Loading {mdl_name}... Please wait.")
                
                if len(mdl_cache) >= 2:
                    cleanup_memory()
                
                mdl_cache[mdl_name] = YOLO(mdl_path)
                
                dummy_size = (320, 320, 3) if "SPACE_ID" in os.environ else (640, 640, 3)
                dummy = np.zeros(dummy_size, dtype=np.uint8)
                mdl_cache[mdl_name].predict(dummy, verbose=False)
                
                memory_after = get_memory_usage()
                st.success(f"✅ {mdl_name} loaded! Memory: {memory_after:.0f}MB")
                
            except ImportError:
                st.error("❌ ultralytics not found. Please check deployment configuration.")
                return None
            except Exception as e:
                st.error(f"❌ Error loading {mdl_name}: {str(e)}")
                
                if "SPACE_ID" in os.environ and mdl_name != 'YOLOv8n (Fastest)':
                    st.info("🤗 Trying lightweight model for HF Spaces...")
                    try:
                        mdl_cache[mdl_name] = YOLO('yolov8n.pt')
                        st.success("✅ Fallback model loaded!")
                    except:
                        st.error("❌ Model loading failed completely.")
                        return None
                else:
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
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border-left: 4px solid #1f77b4;
    }
    
    .status-occupied { color: #d62728; }
    .status-available { color: #2ca02c; }
    
    .sidebar-section {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
    }
    
    .model-info {
        background: #e3f2fd;
        padding: 0.75rem;
        border-radius: 0.5rem;
        border-left: 3px solid #1976d2;
        margin: 0.5rem 0;
    }
    
    .hf-spaces-info {
        background: #f0f8ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #4dabf7;
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

class PDetector:
    def __init__(self, frm_sz: Tuple[int, int] = (1280, 720)):
        self.frm_sz = frm_sz
        self.spaces = []
        self.mdl_name = 'YOLOv8n (Fastest)'
        
        self.OCC_CLR = (0, 0, 255)
        self.AVL_CLR = (0, 255, 0)
        self.DET_CLR = (255, 0, 0)
        self.TXT_CLR = (255, 255, 255)
        
        self.conf_th = 0.3
        self.iou_th = 0.5
        self.sup_cls = ['car', 'truck', 'bus', 'motorcycle', 'bicycle']
        
        self.det_sz_img = (640, 640) if "SPACE_ID" in os.environ else (1280, 1280)
        self.det_sz_vid = (480, 480) if "SPACE_ID" in os.environ else (640, 640)
        self.skip_frm = 5 if "SPACE_ID" in os.environ else 2
        self.last_det = []
        
    def set_model(self, mdl_name: str):
        self.mdl_name = mdl_name
        
    def load_spaces(self, sp_data: List[Dict]) -> None:
        self.spaces = []
        for sp in sp_data:
            space = PSpace(
                id=sp['id'],
                poly=sp['polygon'],
                lbl_pos=tuple(sp['label_position'])
            )
            self.spaces.append(space)
    
    def det_veh_simple(self, frm: np.ndarray, model=None) -> List[Tuple[int, int, int, int, str, float]]:
        if model is None:
            m = get_mdl(self.mdl_name)
            if m is None:
                return []
        else:
            m = model
        
        try:
            if "SPACE_ID" in os.environ:
                h, w = frm.shape[:2]
                if max(h, w) > 640:
                    scale = 640 / max(h, w)
                    new_w, new_h = int(w * scale), int(h * scale)
                    frm = cv2.resize(frm, (new_w, new_h))
            
            results = m.predict(frm, verbose=False, conf=self.conf_th)
            
            if not results or len(results[0].boxes) == 0:
                return []
            
            dets = []
            boxes = results[0].boxes
            
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls_id = int(box.cls[0].cpu().numpy())
                cls_nm = m.names[cls_id]
                
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
        if not self.spaces:
            return frm, {}
        
        rsz_frm = cv2.resize(frm, self.frm_sz)
        dets = self.det_veh_simple(rsz_frm)
        occ = self.chk_occ_simple(dets)
        res_frm = self.draw_res_simple(rsz_frm, occ)
        
        return res_frm, occ
    
    def proc_vid_frm(self, frm: np.ndarray, frm_num: int = 0, model=None) -> Tuple[np.ndarray, Dict[int, Dict]]:
        if not self.spaces:
            return frm, {}
        
        rsz_frm = cv2.resize(frm, self.frm_sz)
        dets = self.det_veh_simple(rsz_frm, model)
        occ = self.chk_occ_simple(dets)
        res_frm = self.draw_res_simple(rsz_frm, occ)
        
        return res_frm, occ

def proc_vid_optimized(vid_path: str, det: PDetector, prog_bar, stat_txt) -> tuple:
    cap = cv2.VideoCapture(vid_path)
    
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    tot_frms = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = det.frm_sz
    
    if "SPACE_ID" in os.environ:
        fps = min(fps, 15)
        max_frames = min(tot_frms, 300)
    else:
        max_frames = tot_frms
    
    out_path = vid_path.replace('.mp4', '_processed.mp4')
    
    # Load model once for entire video
    model = get_mdl(det.mdl_name)
    if model is None:
        raise RuntimeError("Failed to load YOLO model")
    
    fourcc_opts = [
        cv2.VideoWriter_fourcc(*'mp4v'),
        cv2.VideoWriter_fourcc(*'XVID'),
        cv2.VideoWriter_fourcc(*'MJPG'),
        cv2.VideoWriter_fourcc(*'H264')
    ]
    
    out = None
    for fourcc in fourcc_opts:
        try:
            out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
            test_frm = np.zeros((h, w, 3), dtype=np.uint8)
            success = out.write(test_frm)
            if success is not False:
                break
            out.release()
        except:
            continue
    
    if out is None:
        raise RuntimeError("Could not initialize video writer")
    
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
            
            if occ and processed_cnt % 5 == 0:
                occ_cnt = sum(1 for inf in occ.values() if inf['occupied'])
                occ_data.append({
                    'frame': frm_cnt,
                    'timestamp': frm_cnt / fps,
                    'occupied': occ_cnt,
                    'available': len(occ) - occ_cnt
                })
        else:
            res_frm = cv2.resize(frm, det.frm_sz)
        
        out.write(res_frm)
        frm_cnt += 1
        
        if frm_cnt % 10 == 0:
            prog = frm_cnt / max_frames
            prog_bar.progress(min(prog, 1.0))
            stat_txt.text(f"Processing frame {frm_cnt}/{max_frames} ({prog:.1%})")
            
            memory_mb = get_memory_usage()
            if memory_mb > 14000:
                cleanup_memory()
    
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
        
        with st.expander("📖 How to Configure", expanded=False):
            st.markdown("""
            **Configure your parking spaces:**
            
            1. **JSON Format**: Edit the configuration below
            2. **Polygon Points**: Define each space as coordinates [[x1,y1], [x2,y2], ...]
            3. **Label Position**: Where to show the space number
            4. **Preview**: Visual preview with your reference image
            
            **Example:**
            ```json
            {
              "spaces": [
                {
                  "id": 1,
                  "polygon": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
                  "label_position": [x, y]
                }
              ]
            }
            ```
            """)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.write("**📝 Configuration Editor**")
            
            if 'park_cfg' not in st.session_state:
                st.session_state.park_cfg = json.dumps(self.def_cfg, indent=2)
            
            cfg_txt = st.text_area(
                "Edit parking configuration:",
                value=st.session_state.park_cfg,
                height=400,
                help="Edit JSON configuration"
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
                
                st.success(f"✅ Valid config: {len(sps)} spaces")
                st.session_state.park_cfg = cfg_txt
                
                col_a, col_b = st.columns(2)
                
                with col_a:
                    if st.button("🔄 Reset Default"):
                        st.session_state.park_cfg = json.dumps(self.def_cfg, indent=2)
                        st.rerun()
                
                with col_b:
                    st.download_button(
                        label="📥 Download",
                        data=cfg_txt,
                        file_name="parking_config.json",
                        mime="application/json"
                    )
                
                return sps
                
            except json.JSONDecodeError as e:
                st.error(f"❌ JSON Error: {str(e)}")
                return []
            except ValueError as e:
                st.error(f"❌ Config Error: {str(e)}")
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
                    st.error("Fix config errors for preview")
            else:
                st.info("Upload reference image for preview")

def vid_tab():
    st.header("🎬 Video Processing")
    
    if "SPACE_ID" in os.environ:
        st.info("🤗 Video processing is optimized for max 300 frames, 15 FPS")
    
    col1, col2 = st.columns(2)
    with col1:
        skip_frms = st.slider("Frame Skip (higher = faster)", 1, 10, 5 if "SPACE_ID" in os.environ else 2)
        st.session_state.det.skip_frm = skip_frms
    
    with col2:
        max_size = st.selectbox("Video Size Limit (MB)", [10, 25, 50, 100], index=1)
    
    up_vid = st.file_uploader("Upload video", type=['mp4', 'avi', 'mov'])
    
    if up_vid and st.session_state.det.spaces:
        file_size = len(up_vid.getvalue()) / (1024*1024)
        
        if file_size > max_size:
            st.error(f"Video size ({file_size:.1f}MB) exceeds limit ({max_size}MB)")
            return
        
        if st.button("🚀 Process Video", type="primary"):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_f:
                tmp_f.write(up_vid.read())
                vid_path = tmp_f.name
            
            try:
                start_time = time.time()
                
                prog_bar = st.progress(0)
                stat_txt = st.empty()
                
                out_path, occ_data = proc_vid_optimized(
                    vid_path, st.session_state.det, prog_bar, stat_txt
                )
                
                proc_time = time.time() - start_time
                st.success(f"✅ Video processed in {proc_time:.1f}s!")
                
                if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Processed Video")
                        
                        try:
                            st.video(out_path)
                        except Exception as e:
                            st.error(f"Error displaying video: {e}")
                        
                        with open(out_path, 'rb') as f:
                            st.download_button(
                                "📥 Download Processed Video",
                                data=f.read(),
                                file_name="processed_parking_video.mp4",
                                mime="video/mp4"
                            )
                    
                    with col2:
                        st.subheader("Occupancy Analysis")
                        if occ_data:
                            df = pd.DataFrame(occ_data)
                            st.line_chart(df.set_index('timestamp')[['occupied', 'available']])
                            
                            avg_occ = df['occupied'].mean()
                            max_occ = df['occupied'].max()
                            min_occ = df['occupied'].min()
                            
                            st.write("**Video Statistics:**")
                            st.metric("Average Occupied", f"{avg_occ:.1f}")
                            st.metric("Peak Occupancy", f"{max_occ}")
                            st.metric("Minimum Occupancy", f"{min_occ}")
                        else:
                            st.info("No occupancy data generated")
                    
                    if os.path.exists(out_path):
                        os.unlink(out_path)
                else:
                    st.error("Failed to generate output video")
                
            except Exception as e:
                st.error(f"Error processing video: {str(e)}")
            finally:
                if os.path.exists(vid_path):
                    os.unlink(vid_path)
    
    elif up_vid:
        st.warning("⚠️ Configure parking spaces first!")

def main():
    st.markdown('<h1 class="main-header">🚗 Smart Parking Detection System</h1>', unsafe_allow_html=True)
    
    if "SPACE_ID" in os.environ:
        st.markdown("""
        <div class="hf-spaces-info">
            <h3>🤗 Welcome!</h3>
            <p>This parking detection system supports both image and video processing, optimized for 16GB RAM. 
            Features include real-time YOLO detection, configurable parking spaces, and comprehensive analytics.</p>
        </div>
        """, unsafe_allow_html=True)
    
    if 'det' not in st.session_state:
        st.session_state.det = PDetector()
    
    if 'ed' not in st.session_state:
        st.session_state.ed = Editor()
    
    with st.sidebar:
        st.header("⚙️ Settings")
        
        memory_mb = get_memory_usage()
        st.metric("Memory Usage", f"{memory_mb:.0f} MB")
        
        if memory_mb > 10000:
            st.warning("⚠️ High memory usage!")
        
        st.markdown('<div class="model-info">', unsafe_allow_html=True)
        st.write("**🧠 AI Model**")
        
        mdl_choice = st.selectbox(
            "YOLO Model:",
            list(YOLO_MODELS.keys()),
            index=0,
            help="YOLOv8n recommended for HF Spaces"
        )
        
        if mdl_choice != st.session_state.det.mdl_name:
            st.session_state.det.set_model(mdl_choice)
            st.info(f"Switched to {mdl_choice}")
        
        model_info = {
            'YOLOv8n (Fastest)': '⚡ Best for HF Spaces',
            'YOLOv8s (Balanced)': '⚖️ Good balance',
            'YOLOv8m (Better Accuracy)': '🎯 Higher accuracy'
        }
        
        st.info(model_info.get(mdl_choice, 'YOLO model'))
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.write("**🎛️ Detection**")
        
        conf = st.slider("Confidence", 0.1, 0.9, 0.3, 0.05)
        st.session_state.det.conf_th = conf
        
        iou_th = st.slider("IoU Threshold", 0.1, 0.9, 0.5, 0.05)
        st.session_state.det.iou_th = iou_th
        
        st.write("**🎬 Video Settings**")
        
        if "SPACE_ID" in os.environ:
            st.info("Video limited to 300 frames on HF Spaces")
        else:
            st.info("Full video processing available")
        
        st.write("**🤖 Model Management**")
        
        if st.button("🔄 Reload Model"):
            if mdl_choice in mdl_cache:
                del mdl_cache[mdl_choice]
            get_mdl(mdl_choice)
        
        if st.button("🗑️ Clear Cache"):
            mdl_cache.clear()
            cleanup_memory()
            st.success("Cache cleared!")

    tab1, tab2, tab3, tab4 = st.tabs(["🔧 Setup", "📷 Image Detection", "🎬 Video Processing", "📊 Analytics"])
    
    with tab1:
        st.header("🔧 Configuration")
        
        ref_img = st.file_uploader(
            "Upload reference image", 
            type=['jpg', 'jpeg', 'png'],
            help="Upload parking lot image to configure spaces"
        )
        
        img_arr = None
        if ref_img:
            img = Image.open(ref_img)
            img_arr = np.array(img)
            if len(img_arr.shape) == 3:
                img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
        
        sps = st.session_state.ed.render_ed(img_arr)
        
        if sps:
            st.session_state.det.load_spaces(sps)
            st.success(f"✅ Loaded {len(sps)} parking spaces")
    
    with tab2:
        st.header("📷 Image Detection")
        
        col_info1, col_info2 = st.columns(2)
        with col_info1:
            st.info(f"🧠 Model: **{st.session_state.det.mdl_name}**")
        with col_info2:
            st.info(f"📊 Memory: **{get_memory_usage():.0f}MB**")
        
        up_img = st.file_uploader(
            "Upload image for detection", 
            type=['jpg', 'jpeg', 'png'], 
            key="img_up"
        )
        
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
                
                if st.session_state.det.spaces:
                    with st.spinner(f"Processing with {st.session_state.det.mdl_name}..."):
                        start_time = time.time()
                        res_frm, occ = st.session_state.det.proc_img(img_arr)
                        proc_time = time.time() - start_time
                        
                        res_rgb = cv2.cvtColor(res_frm, cv2.COLOR_BGR2RGB)
                        st.image(res_rgb, use_container_width=True)
                        
                        st.caption(f"⏱️ Processed in {proc_time:.2f}s")
                        
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
                            
                            st.subheader("Space Details")
                            for sp_id, inf in occ.items():
                                if inf['occupied'] and inf['vehicle']:
                                    veh = inf['vehicle']
                                    st.write(f"🔴 **Space {sp_id}**: {veh['class'].title()} "
                                           f"({veh['confidence']:.2f})")
                                else:
                                    st.write(f"🟢 **Space {sp_id}**: Available")
                        else:
                            st.info("No detection results")
                else:
                    st.warning("⚠️ Configure parking spaces first in Setup tab!")
    
    with tab3:
        vid_tab()
    
    with tab4:
        st.header("📊 Analytics")
        
        if st.session_state.det.spaces:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Model Performance")
                
                perf_data = {
                    'Model': ['YOLOv8n', 'YOLOv8s', 'YOLOv8m'],
                    'Speed (FPS)': [150, 120, 80],
                    'Accuracy': [37.3, 44.9, 50.2],
                    'Size (MB)': [6.2, 21.5, 49.7]
                }
                
                df = pd.DataFrame(perf_data)
                st.dataframe(df, use_container_width=True)
            
            with col2:
                st.subheader("System Status")
                
                system_info = {
                    "Platform": "Hugging Face Spaces" if "SPACE_ID" in os.environ else "Local",
                    "Model": st.session_state.det.mdl_name,
                    "Confidence": st.session_state.det.conf_th,
                    "IoU Threshold": st.session_state.det.iou_th,
                    "Memory Usage": f"{get_memory_usage():.0f}MB",
                    "Video Skip Frames": st.session_state.det.skip_frm
                }
                
                st.json(system_info)
            
            st.subheader("Sample Analytics")
            
            hours = list(range(24))
            occupancy = np.random.uniform(0.2, 0.9, 24)
            
            df_hourly = pd.DataFrame({
                'Hour': hours,
                'Occupancy': occupancy
            })
            
            st.write("**Hourly Occupancy Pattern**")
            st.line_chart(df_hourly.set_index('Hour'))
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Peak", f"{occupancy.max():.1%}")
            with col_b:
                st.metric("Average", f"{occupancy.mean():.1%}")
            with col_c:
                st.metric("Minimum", f"{occupancy.min():.1%}")
            
            st.subheader("Video Processing Stats")
            
            if "SPACE_ID" in os.environ:
                st.info("🤗 HF Spaces: Max 300 frames, 15 FPS, optimized processing")
            else:
                st.info("💻 Local: Full video processing, all frames")
                
        else:
            st.warning("Configure parking spaces first!")
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p><strong>🚗 Smart Parking Detection System</strong></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()