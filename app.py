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
    'YOLOv8x (Best Accuracy)': 'yolov8x.pt',
    'YOLOv9c (Latest)': 'yolov9c.pt',
    'YOLOv9e (Premium)': 'yolov9e.pt',
    'YOLOv10n (Efficient)': 'yolov10n.pt',
    'YOLOv10s (Stable)': 'yolov10s.pt',
    'YOLOv10m (Advanced)': 'yolov10m.pt'
}

def get_mdl(mdl_name: str = 'YOLOv8s (Balanced)'):
    global mdl_cache
    with mdl_lock:
        mdl_path = YOLO_MODELS.get(mdl_name, 'yolov8s.pt')
        
        if mdl_name not in mdl_cache:
            try:
                from ultralytics import YOLO
                st.info(f"Loading {mdl_name}... Please wait.")
                mdl_cache[mdl_name] = YOLO(mdl_path)
                
                dummy = np.zeros((640, 640, 3), dtype=np.uint8)
                mdl_cache[mdl_name].predict(dummy, verbose=False)
                st.success(f"✅ {mdl_name} loaded successfully!")
                
            except ImportError:
                st.error("❌ ultralytics not found. Install with: `pip install ultralytics`")
                return None
            except Exception as e:
                st.error(f"❌ Error loading {mdl_name}: {str(e)}")
                st.info("Trying to download model automatically...")
                return None
                
    return mdl_cache.get(mdl_name)

st.set_page_config(
    page_title="Smart Parking Detection System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color:
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(90deg,
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border-left: 4px solid
    }
    
    .status-occupied { color:
    .status-available { color:
    
    .sidebar-section {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
        background-color:
        border: 1px solid
    }
    
    .model-info {
        background:
        padding: 0.75rem;
        border-radius: 0.5rem;
        border-left: 3px solid
        margin: 0.5rem 0;
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
        self.mdl_name = 'YOLOv8s (Balanced)'
        
        self.OCC_CLR = (0, 0, 255)
        self.AVL_CLR = (0, 255, 0)
        self.DET_CLR = (255, 0, 0)
        self.TXT_CLR = (255, 255, 255)
        
        self.conf_th = 0.3
        self.iou_th = 0.5
        self.sup_cls = ['car', 'truck', 'bus', 'motorcycle', 'bicycle']
        
        self.det_sz_img = (1280, 1280)
        self.det_sz_vid = (640, 640)
        self.skip_frm = 2
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
    
    def det_veh_simple(self, frm: np.ndarray) -> List[Tuple[int, int, int, int, str, float]]:
        m = get_mdl(self.mdl_name)
        if m is None:
            return []
        
        try:

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
                

                if cls_nm.lower() == 'car':
                    dets.append((x1, y1, x2, y2, cls_nm, float(conf)))
            
            return dets
            
        except Exception as e:
            log.error(f"Detection error: {str(e)}")
            return []
    
    def det_veh_cached(self, frm: np.ndarray, use_cache: bool = False) -> List[Tuple[int, int, int, int, str, float]]:
        if use_cache and self.last_det:
            return self.last_det
            
        dets = self.det_veh_enhanced(frm, high_res=False)
        self.last_det = dets
        return dets
    
    def chk_occ_simple(self, dets: List[Tuple[int, int, int, int, str, float]]) -> Dict[int, Dict]:
        occ = {}
        
        for sp in self.spaces:
            occupied = False
            veh = None
            
            for x1, y1, x2, y2, cls_nm, conf in dets:

                if cls_nm.lower() == 'car':

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
    
    def calc_overlap_area(self, poly_arr: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
        try:
            x1, y1, x2, y2 = bbox
            bbox_poly = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
            
            intersection = cv2.intersectConvexConvex(poly_arr, bbox_poly)[1]
            if intersection is not None and len(intersection) > 2:
                return cv2.contourArea(intersection)
            return 0
        except:
            return 0
    
    def draw_res_simple(self, frm: np.ndarray, occ: Dict[int, Dict]) -> np.ndarray:
        res_frm = frm.copy()
        
        for sp in self.spaces:
            sp_inf = occ[sp.id]
            

            if sp_inf['occupied']:
                cv2.polylines(res_frm, [sp.poly_arr], True, (0, 0, 255), 2)
                cv2.putText(res_frm, str(sp.id), sp.lbl_pos, 
                           cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
                

                if sp_inf['vehicle']:
                    veh = sp_inf['vehicle']
                    x1, y1, x2, y2 = veh['bbox']
                    cx, cy = veh['center']
                    
                    cv2.rectangle(res_frm, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(res_frm, (cx, cy), 3, (0, 0, 255), -1)
                    cv2.putText(res_frm, veh['class'], (x1, y1), 
                               cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
            else:
                cv2.polylines(res_frm, [sp.poly_arr], True, (0, 255, 0), 2)
                cv2.putText(res_frm, str(sp.id), sp.lbl_pos, 
                           cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
        

        occ_cnt = sum(1 for inf in occ.values() if inf['occupied'])
        tot_sp = len(self.spaces)
        avl_sp = tot_sp - occ_cnt
        

        cv2.putText(res_frm, str(avl_sp), (292, 404), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2)
        

        summary = f"Occupied: {occ_cnt}/{tot_sp}"
        cv2.rectangle(res_frm, (10, 10), (300, 40), (0, 0, 0), -1)
        cv2.putText(res_frm, summary, (15, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return res_frm
    
    def proc_img(self, frm: np.ndarray) -> Tuple[np.ndarray, Dict[int, Dict]]:
        if not self.spaces:
            return frm, {}
        

        rsz_frm = cv2.resize(frm, (1280, 720))
        

        dets = self.det_veh_simple(rsz_frm)
        

        occ = self.chk_occ_simple(dets)
        

        res_frm = self.draw_res_simple(rsz_frm, occ)
        
        return res_frm, occ
    
    def proc_vid_frm(self, frm: np.ndarray, frm_num: int = 0) -> Tuple[np.ndarray, Dict[int, Dict]]:
        if not self.spaces:
            return frm, {}
        

        rsz_frm = cv2.resize(frm, (1280, 720))
        

        dets = self.det_veh_simple(rsz_frm)
        

        occ = self.chk_occ_simple(dets)
        

        res_frm = self.draw_res_simple(rsz_frm, occ)
        
        return res_frm, occ

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
        st.subheader("🛠️ YBAT-Style Parking Configuration")
        
        with st.expander("📖 Configuration Instructions", expanded=False):
            st.markdown("""
            **How to configure parking spaces:**
            
            1. **JSON Format**: Edit the configuration in JSON format below
            2. **Polygon Points**: Define each parking space as a polygon with (x,y) coordinates
            3. **Label Position**: Set where the space ID label appears
            4. **Validation**: Real-time validation shows errors immediately
            5. **Preview**: Visual preview shows your configuration on the image
            
            **Example Format:**
            ```json
            {
              "spaces": [
                {
                  "id": 1,
                  "polygon": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
                  "label_position": [label_x, label_y]
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
                help="Edit the JSON configuration for parking spaces"
            )
            
            try:
                cfg_data = json.loads(cfg_txt)
                
                if 'spaces' not in cfg_data:
                    raise ValueError("Configuration must have 'spaces' key")
                
                sps = cfg_data['spaces']
                if not isinstance(sps, list):
                    raise ValueError("'spaces' must be a list")
                
                for i, sp in enumerate(sps):
                    req_keys = ['id', 'polygon', 'label_position']
                    for key in req_keys:
                        if key not in sp:
                            raise ValueError(f"Space {i} missing required key: {key}")
                    
                    if len(sp['polygon']) < 3:
                        raise ValueError(f"Space {sp['id']} polygon must have at least 3 points")
                
                st.success(f"✅ Valid configuration with {len(sps)} spaces")
                st.session_state.park_cfg = cfg_txt
                
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    if st.button("💾 Save Config", type="primary"):
                        with open("parking_config.json", "w") as f:
                            json.dump(cfg_data, f, indent=2)
                        st.success("Configuration saved!")
                
                with col_b:
                    if st.button("🔄 Reset Default"):
                        st.session_state.park_cfg = json.dumps(self.def_cfg, indent=2)
                        st.rerun()
                
                with col_c:
                    st.download_button(
                        label="📥 Download",
                        data=cfg_txt,
                        file_name="parking_config.json",
                        mime="application/json"
                    )
                
                up_cfg = st.file_uploader("📤 Upload Configuration", type=['json'])
                if up_cfg:
                    try:
                        up_data = json.load(up_cfg)
                        st.session_state.park_cfg = json.dumps(up_data, indent=2)
                        st.success("Configuration uploaded!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error uploading config: {str(e)}")
                
                return sps
                
            except json.JSONDecodeError as e:
                st.error(f"❌ JSON Syntax Error: {str(e)}")
                return []
            except ValueError as e:
                st.error(f"❌ Configuration Error: {str(e)}")
                return []
        
        with col2:
            st.write("**👁️ Configuration Preview**")
            
            if img_arr is not None:
                prev_img = img_arr.copy()
                
                try:
                    cfg_data = json.loads(cfg_txt)
                    sps = cfg_data.get('spaces', [])
                    
                    for sp in sps:
                        poly = np.array(sp['polygon'], dtype=np.int32)
                        cv2.polylines(prev_img, [poly], True, (0, 255, 0), 2)
                        
                        lbl_pos = tuple(sp['label_position'])
                        cv2.putText(prev_img, f"Space {sp['id']}", lbl_pos,
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    prev_rgb = cv2.cvtColor(prev_img, cv2.COLOR_BGR2RGB)
                    st.image(prev_rgb, caption="Configuration Preview", use_container_width=True)
                    
                except:
                    st.error("Cannot preview - fix configuration errors first")
            else:
                st.info("Upload an image to see configuration preview")

def proc_vid_fixed(vid_path: str, det: PDetector, prog_bar, stat_txt) -> tuple:
    cap = cv2.VideoCapture(vid_path)
    
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    tot_frms = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = det.frm_sz
    
    out_path = vid_path.replace('.mp4', '_processed.mp4')
    
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
        raise RuntimeError("Could not initialize video writer with any codec")
    
    frm_cnt = 0
    occ_data = []
    
    det.last_det = []
    
    while True:
        ret, frm = cap.read()
        if not ret:
            break
        
        res_frm, occ = det.proc_vid_frm(frm, frm_cnt)
        out.write(res_frm)
        
        if occ and frm_cnt % 10 == 0:
            occ_cnt = sum(1 for inf in occ.values() if inf['occupied'])
            occ_data.append({
                'frame': frm_cnt,
                'timestamp': frm_cnt / fps,
                'occupied': occ_cnt,
                'available': len(occ) - occ_cnt
            })
        
        frm_cnt += 1
        
        if frm_cnt % 10 == 0:
            prog = frm_cnt / tot_frms if tot_frms > 0 else 0
            prog_bar.progress(min(prog, 1.0))
            stat_txt.text(f"Processing frame {frm_cnt}/{tot_frms} ({prog:.1%})")
    
    cap.release()
    out.release()
    
    return out_path, occ_data

def vid_tab():
    st.header("🎬 Video Detection")
    
    col1, col2 = st.columns(2)
    with col1:
        skip_frms = st.slider("Frame Skip (higher = faster)", 1, 10, 2)
        st.session_state.det.skip_frm = skip_frms
    
    with col2:
        det_sz = st.selectbox("Video Detection Size", 
                            [(480, 480), (640, 640), (800, 800)], 
                            index=1)
        st.session_state.det.det_sz_vid = det_sz
    
    up_vid = st.file_uploader("Upload video", type=['mp4', 'avi', 'mov'])
    
    if up_vid and st.session_state.det.spaces:
        if st.button("🚀 Process Video", type="primary"):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_f:
                tmp_f.write(up_vid.read())
                vid_path = tmp_f.name
            
            try:
                st_time = time.time()
                
                prog_bar = st.progress(0)
                stat_txt = st.empty()
                
                out_path, occ_data = proc_vid_fixed(
                    vid_path, st.session_state.det, prog_bar, stat_txt
                )
                
                proc_time = time.time() - st_time
                st.success(f"✅ Video processing completed in {proc_time:.1f} seconds!")
                
                if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Processed Video")
                        
                        try:
                            st.video(out_path)
                        except Exception as e:
                            st.error(f"Error displaying video: {e}")
                            st.info("Try downloading the video file instead.")
                        
                        with open(out_path, 'rb') as f:
                            st.download_button(
                                "📥 Download Processed Video",
                                data=f.read(),
                                file_name="processed_video.mp4",
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
                            st.write(f"• Average Occupied: {avg_occ:.1f}")
                            st.write(f"• Peak Occupancy: {max_occ}")
                            st.write(f"• Minimum Occupancy: {min_occ}")
                        else:
                            st.info("No occupancy data generated")
                    
                    if os.path.exists(out_path):
                        os.unlink(out_path)
                else:
                    st.error("Failed to generate output video")
                
            except Exception as e:
                st.error(f"Error processing video: {str(e)}")
                st.error("Please try a different video format or check video integrity")
            finally:
                if os.path.exists(vid_path):
                    os.unlink(vid_path)
    
    elif up_vid:
        st.warning("⚠️ Configure parking spaces first!")

def main():
    st.markdown('<h1 class="main-header">🚗 Smart Parking Detection System</h1>', unsafe_allow_html=True)
    
    if 'det' not in st.session_state:
        st.session_state.det = PDetector()
    
    if 'ed' not in st.session_state:
        st.session_state.ed = Editor()
    
    with st.sidebar:
        st.header("⚙️ Detection Settings")
        
        st.markdown('<div class="model-info">', unsafe_allow_html=True)
        st.write("**🧠 AI Model Selection**")
        
        mdl_choice = st.selectbox(
            "Choose YOLO Model:",
            list(YOLO_MODELS.keys()),
            index=1,
            help="Higher accuracy models are slower but more precise"
        )
        
        if mdl_choice != st.session_state.det.mdl_name:
            st.session_state.det.set_model(mdl_choice)
            st.info(f"Model changed to {mdl_choice}")
        
        model_info = {
            'YOLOv8n (Fastest)': '⚡ Fastest, Good for real-time',
            'YOLOv8s (Balanced)': '⚖️ Best speed/accuracy balance',
            'YOLOv8m (Better Accuracy)': '🎯 Higher accuracy, slower',
            'YOLOv8l (High Accuracy)': '🏆 High accuracy, more compute',
            'YOLOv8x (Best Accuracy)': '👑 Best accuracy, slowest',
            'YOLOv9c (Latest)': '🚀 Latest technology',
            'YOLOv9e (Premium)': '💎 Premium performance',
            'YOLOv10n (Efficient)': '⚡ New efficient model',
            'YOLOv10s (Stable)': '🛡️ Stable and reliable',
            'YOLOv10m (Advanced)': '🔬 Advanced detection'
        }
        
        st.info(model_info.get(mdl_choice, 'Advanced YOLO model'))
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.write("**🎛️ Detection Parameters**")
        
        conf = st.slider("Detection Confidence", 0.1, 0.9, 0.3, 0.05, 
                        help="Lower = more detections, Higher = more confident detections")
        st.session_state.det.conf_th = conf
        
        iou_th = st.slider("IoU Threshold", 0.1, 0.9, 0.5, 0.05,
                          help="Controls overlap filtering - lower removes more overlaps")
        st.session_state.det.iou_th = iou_th
        
        st.write("**📐 Frame Settings**")
        frm_w = st.selectbox("Frame Width", [640, 1280, 1920], index=1)
        frm_h = st.selectbox("Frame Height", [480, 720, 1080], index=1)
        st.session_state.det.frm_sz = (frm_w, frm_h)
        
        img_det_sz = st.selectbox("Image Detection Size", 
                                 [(640, 640), (1280, 1280), (1920, 1920)], 
                                 index=1,
                                 help="Higher = better accuracy, slower processing")
        st.session_state.det.det_sz_img = img_det_sz
        
        st.write("**🤖 Model Management**")
        if st.button("🔄 Load/Reload Model"):
            if mdl_choice in mdl_cache:
                del mdl_cache[mdl_choice]
            get_mdl(mdl_choice)
        
        if st.button("🗑️ Clear All Models"):
            mdl_cache.clear()
            st.success("All models cleared from memory")

    tab1, tab2, tab3, tab4 = st.tabs(["🔧 Configuration", "📷 Image Detection", "🎬 Video Detection", "📊 Analytics"])
    
    with tab1:
        st.header("🔧 YBAT-Style Configuration")
        
        ref_img = st.file_uploader(
            "Upload reference image (optional)", 
            type=['jpg', 'jpeg', 'png'],
            help="Upload an image to preview your configuration"
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
            st.success(f"✅ Loaded {len(sps)} parking spaces into detector")
    
    with tab2:
        st.header("📷 Enhanced Image Detection")
        
        col_info1, col_info2 = st.columns(2)
        with col_info1:
            st.info(f"🧠 Using: **{st.session_state.det.mdl_name}**")
        with col_info2:
            st.info(f"🎯 Detection Size: **{st.session_state.det.det_sz_img[0]}x{st.session_state.det.det_sz_img[1]}**")
        
        if st.button("🔄 Clear Cache", help="Clear cached detections for fresh image processing"):
            st.session_state.det.last_det = []
            st.success("Cache cleared!")
        
        up_img = st.file_uploader("Upload image for detection", type=['jpg', 'jpeg', 'png'], key="img_up")
        
        if up_img:
            img = Image.open(up_img)
            img_arr = np.array(img)
            if len(img_arr.shape) == 3:
                img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
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
                        
                        st.caption(f"⏱️ Processing time: {proc_time:.2f}s")
                        
                        if occ:
                            occ_cnt = sum(1 for inf in occ.values() if inf['occupied'])
                            tot = len(occ)
                            avl = tot - occ_cnt
                            occ_rate = (occ_cnt / tot * 100) if tot > 0 else 0
                            
                            col_a, col_b, col_c, col_d = st.columns(4)
                            with col_a:
                                st.metric("Total Spaces", tot)
                            with col_b:
                                st.metric("Available", avl, delta=None)
                            with col_c:
                                st.metric("Occupied", occ_cnt, delta=None)
                            with col_d:
                                st.metric("Occupancy Rate", f"{occ_rate:.1f}%")
                            
                            st.subheader("Detailed Space Analysis")
                            for sp_id, inf in occ.items():
                                if inf['occupied'] and inf['vehicle']:
                                    veh = inf['vehicle']
                                    overlap = veh.get('overlap_ratio', 0) * 100
                                    st.write(f"🔴 **Space {sp_id}**: {veh['class'].title()} detected "
                                           f"(Confidence: {veh['confidence']:.2f}, Overlap: {overlap:.1f}%)")
                                else:
                                    st.write(f"🟢 **Space {sp_id}**: Available")
                        else:
                            st.info("No occupancy data generated")
                else:
                    st.warning("⚠️ Configure parking spaces first!")
    
    with tab3:
        vid_tab()
    
    with tab4:
        st.header("📊 Analytics Dashboard")
        
        if st.session_state.det.spaces:
            st.subheader("📈 Performance Analytics")
            
            col_perf1, col_perf2 = st.columns(2)
            
            with col_perf1:
                st.write("**🧠 Model Comparison**")
                
                models_perf = {
                    'Model': ['YOLOv8n', 'YOLOv8s', 'YOLOv8m', 'YOLOv8l', 'YOLOv8x'],
                    'Speed (FPS)': [150, 120, 80, 45, 25],
                    'Accuracy (mAP)': [37.3, 44.9, 50.2, 52.9, 53.9],
                    'Size (MB)': [6.2, 21.5, 49.7, 83.7, 131.4]
                }
                
                df_models = pd.DataFrame(models_perf)
                st.dataframe(df_models, use_container_width=True)
                
            with col_perf2:
                st.write("**⚙️ Current Configuration**")
                st.json({
                    "Model": st.session_state.det.mdl_name,
                    "Confidence Threshold": st.session_state.det.conf_th,
                    "IoU Threshold": st.session_state.det.iou_th,
                    "Image Detection Size": f"{st.session_state.det.det_sz_img[0]}x{st.session_state.det.det_sz_img[1]}",
                    "Video Detection Size": f"{st.session_state.det.det_sz_vid[0]}x{st.session_state.det.det_sz_vid[1]}",
                    "Frame Skip": st.session_state.det.skip_frm
                })
            
            st.subheader("📈 Sample Analytics")
            
            hrs = list(range(24))
            occ_rates = np.random.uniform(0.2, 0.9, 24)
            
            df_anal = pd.DataFrame({
                'Hour': hrs,
                'Occupancy_Rate': occ_rates
            })
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Hourly Occupancy Pattern**")
                st.line_chart(df_anal.set_index('Hour'))
            
            with col2:
                st.write("**Key Metrics**")
                st.metric("Peak Occupancy", f"{occ_rates.max():.1%}")
                st.metric("Average Occupancy", f"{occ_rates.mean():.1%}")
                st.metric("Lowest Occupancy", f"{occ_rates.min():.1%}")
            
            st.subheader("📅 Weekly Pattern")
            days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            wk_data = np.random.uniform(0.3, 0.8, 7)
            
            df_wk = pd.DataFrame({
                'Day': days,
                'Average_Occupancy': wk_data
            })
            
            st.bar_chart(df_wk.set_index('Day'))
            
        else:
            st.warning("Configure parking spaces to view analytics!")
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color:
        <p><strong>🚗 Smart Parking Detection System v2.0</strong> - Enhanced with Multiple AI Models</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()