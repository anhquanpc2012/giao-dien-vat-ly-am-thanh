import streamlit as st
import numpy as np
import scipy.signal as signal
from scipy.signal import butter, sosfilt
import matplotlib.pyplot as plt
import soundfile as sf
import io
from audio_recorder_streamlit import audio_recorder

# 1. Cấu hình giao diện chạy mượt mà trên trình duyệt điện thoại
st.set_page_config(page_title="Physics Audio Analyzer Pro", layout="wide")

st.title("🎵 Hệ Thống Vật Lý Phân Tích Âm Thanh Pro")
st.caption("Ứng dụng chạy trực tuyến vĩnh viễn - Hỗ trợ Micro thời gian thực trên thiết bị di động")

# --- LỚP XỬ LÝ THUẬT TOÁN VẬT LÝ (TÍCH HỢP TỪ FILE MP-571651.PY CỦA BẠN) ---
class PhysicsAudioAnalyzerPipeline:
    def __init__(self, sampling_rate=44100):
        self.sr = sampling_rate
        self.y = None
        self.hop_length_stft = 512
        
    def generate_synthetic_signal(self, base_freq=220.0, duration=3.0, noise_factor=0.1):
        t = np.linspace(0, duration, int(self.sr * duration), endpoint=False)
        fundamental = 0.5 * np.sin(2 * np.pi * base_freq * t)
        overtone = 0.2 * np.sin(2 * np.pi * (base_freq * 2) * t)
        white_noise = noise_factor * np.random.normal(0, 1, len(t))
        self.y = fundamental + overtone + white_noise
        return self.y

    def analyze_time_domain(self, frame_length=2048, hop_length=512):
        if self.y is None: 
            raise ValueError("Chưa có dữ liệu âm thanh!")
            
        zcr_vector = []
        num_frames = 1 + int((len(self.y) - frame_length) / hop_length)
        
        for i in range(num_frames):
            start = i * hop_length
            end = start + frame_length
            frame = self.y[start:end]
            if len(frame) < frame_length: continue
            
            sign_changes = np.sum(np.abs(np.diff(np.sign(frame)))) / 2
            zcr_vector.append(sign_changes / frame_length)
            
        return np.array(zcr_vector)

    def execute_fft_spectrum(self):
        N = len(self.y)
        fft_data = np.fft.fft(self.y)
        frequencies = np.fft.fftfreq(N, d=1/self.sr)[:N // 2]
        magnitude_spectrum = np.abs(fft_data)[:N // 2]
        return frequencies, magnitude_spectrum

    def extract_advanced_features(self):
        nperseg = 2048
        noverlap = 1536
        self.hop_length_stft = nperseg - noverlap 
        
        f, t, Zxx = signal.stft(self.y, fs=self.sr, window='hann', nperseg=nperseg, noverlap=noverlap)
        spectrogram_db = 20 * np.log10(np.abs(Zxx) + 1e-6)
        
        mfcc_features = np.zeros((13, len(t))) 
        for i in range(13):
            mfcc_features[i, :] = np.sin(t * (i + 1)) * (13 - i)
            
        return f, t, spectrogram_db, mfcc_features

    def design_lowpass_butterworth(self, cutoff_freq=600.0, order=5):
        nyquist = 0.5 * self.sr
        normal_cutoff = cutoff_freq / nyquist
        sos = butter(order, normal_cutoff, btype='low', analog=False, output='sos')
        y_filtered = sosfilt(sos, self.y)
        return y_filtered

    def estimate_pitch_autocorr(self, frame_size=2048):
        start = len(self.y) // 2
        if start + frame_size > len(self.y):
            start = 0
        frame = self.y[start : start + frame_size]
        if len(frame) < 128: return 0.0
        
        autocorr = np.correlate(frame, frame, mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        
        min_lag = int(self.sr / 500)
        max_lag = int(self.sr / 60)
        if max_lag > len(autocorr): max_lag = len(autocorr)
        if min_lag >= max_lag: return 0.0
        
        chosen_lag = min_lag + np.argmax(autocorr[min_lag:max_lag])
        return self.sr / chosen_lag if chosen_lag > 0 else 0.0

# --- THIẾT KẾ GIAO DIỆN CHỌN NGUỒN ÂM THANH ---
tab1, tab2 = st.tabs(["🎙️ Thu âm từ Micro thiết bị", "📂 Tải file nhạc sẵn có lên"])

audio_data_bytes = None

with tab1:
    st.write("Nhấp vào biểu tượng Micro bên dưới để bắt đầu nói/huýt sáo, nhấp lại để dừng và phân tích:")
    # Chạy chế độ mặc định nguyên bản của thư viện để an toàn tuyệt đối
    audio_data_bytes = audio_recorder()

with tab2:
    uploaded_file = st.file_uploader("Chọn file âm thanh (.wav, .flac, .mp3)", type=["wav", "flac", "mp3"])
    if uploaded_file is not None:
        audio_data_bytes = uploaded_file.read()

# --- KHỞI CHẠY PIPELINE VÀ HIỂN THỊ ĐỒ THỊ ---
if audio_data_bytes is not None:
    with st.spinner("Hệ thống vật lý đang tính toán ma trận sóng âm..."):
        try:
            # 1. Đọc dữ liệu nhị phân từ bộ nhớ RAM
            data, samplerate = sf.read(io.BytesIO(audio_data_bytes))
            
            # Khởi tạo đối tượng phân tích
            pipeline = PhysicsAudioAnalyzerPipeline(sampling_rate=samplerate)
            pipeline.y = data
            if len(pipeline.y.shape) > 1:
                pipeline.y = pipeline.y.mean(axis=1) # Ép về Mono nếu là bài hát Stereo
            
            # 2. Thực thi chuỗi thuật toán xử lý tín hiệu số
            zcr_features = pipeline.analyze_time_domain()
            frequencies, fft_magnitude = pipeline.execute_fft_spectrum()
            spec_f, spec_t, spectrogram_db, mfcc_features = pipeline.extract_advanced_features()
            filtered_audio = pipeline.design_lowpass_butterworth(cutoff_freq=600.0)
            estimated_f0 = pipeline.estimate_pitch_autocorr()
            
            # 3. Hiển thị bảng số liệu trực quan hóa nhanh
            col1, col2 = st.columns(2)
            col1.metric("Tần số lấy mẫu thực tế (Fs)", f"{samplerate} Hz")
            col2.metric("Cao độ cơ bản tìm được (Pitch F0)", f"{estimated_f0:.2f} Hz" if estimated_f0 > 0 else "Không xác định")

            # 4. Tạo bố cục đồ thị dọc tối ưu riêng cho màn hình điện thoại
            fig, axes = plt.subplots(5, 1, figsize=(10, 16))
            fig.suptitle(f"Biểu Đồ Tổng Hợp Kiến Thức - F0: {estimated_f0:.2f} Hz", fontsize=14, fontweight='bold')

            # Đồ thị 1: Dạng sóng
            t_signal = np.linspace(0, len(pipeline.y)/samplerate, len(pipeline.y))
            sample_limit = min(3000, len(pipeline.y))
            axes[0].plot(t_signal[:sample_limit], pipeline.y[:sample_limit], color='blue', linewidth=0.7)
            axes[0].set_title("1. Dạng sóng Biên độ Miền thời gian (Phóng to đoạn đầu)")
            axes[0].set_xlabel("Thời gian (s)")
            
            # Đồ thị 2: ZCR
            axes[1].plot(zcr_features, color='green')
            axes[1].set_title("2. Tốc độ qua điểm không (Zero-Crossing Rate)")
            
            # Đồ thị 3: Phổ FFT
            axes[2].plot(frequencies, fft_magnitude, color='red', linewidth=0.8)
            axes[2].set_xlim(0, 1500)
            axes[2].set_title("3. Phổ Tần số Biên độ (FFT Spectrum)")
            axes[2].set_xlabel("Tần số (Hz)")
            
            # Đồ thị 4: Spectrogram
            pcm = axes[3].pcolormesh(spec_t, spec_f, spectrogram_db, shading='gouraud', cmap='magma')
            axes[3].set_ylim(0, 1500)
            axes[3].set_title("4. Đồ thị Quang phổ (Short-Time Fourier Spectrogram)")
            fig.colorbar(pcm, ax=axes[3], format='%+2.0f dB')
            
            # Đồ thị 5: Hệ số phi tuyến giả lập
            extent = [spec_t[0], spec_t[-1], 0, 13]
            img2 = axes[4].imshow(mfcc_features, aspect='auto', origin='lower', cmap='viridis', extent=extent)
            axes[4].set_title("5. Ma trận Đặc trưng Giả lập (13 Hệ số Phi tuyến)")
            fig.colorbar(img2, ax=axes[4])

            plt.tight_layout()
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Định dạng âm thanh từ thiết bị chưa tương thích: {e}")
else:
    st.info("💡 Hệ thống đã sẵn sàng. Hãy bấm nút biểu tượng Micro để ghi âm trực tiếp hoặc tải file lên!")