# Tổng Quan Nghiên Cứu Text-to-Speech

## Giới thiệu

Text-to-Speech (TTS) là công nghệ chuyển đổi văn bản thành giọng nói tự nhiên, đóng vai trò quan trọng trong nhiều ứng dụng như trợ lý ảo, hệ thống đọc sách tự động, và công nghệ hỗ trợ người khuyết tật. Bài viết này tổng quan về các phương pháp TTS từ truyền thống đến hiện đại, phân tích kiến trúc và ứng dụng của chúng.

---

## 1. Phát triển lịch sử của Text-to-Speech

### 1.1. Giai đoạn đầu (trước năm 2000)
- **Concatenative Synthesis:** Ghép nối các đoạn âm thanh được ghi âm trước
- **Formant Synthesis:** Tạo âm thanh dựa trên mô hình toán học
- **Articulatory Synthesis:** Mô phỏng quá trình sinh học của hệ thống phát âm

**Hạn chế:**
- Giọng nói thiếu tự nhiên, máy móc
- Khó điều chỉnh tông giọng và ngữ điệu
- Yêu cầu nhiều công sức thu thập và xử lý dữ liệu

### 1.2. Kỷ nguyên Deep Learning (2013-2016)
- **WaveNet (DeepMind, 2016):** Mô hình sinh âm thanh từng mẫu một
  - Sử dụng mạng tích chập giãn nở
  - Cải thiện đáng kể chất lượng giọng nói
  - Nhược điểm: Vì phải sinh từng mẫu âm thanh một cách tuần tự, nên tốc độ rất chậm, khó dùng trong thời gian thực.

- **Tacotron (Google, 2017):** End-to-end TTS biến văn bản → giọng nói
  - Gồm text encoder (xử lý văn bản) và audio decoder (tạo âm thanh).
  - Sử dụng cơ chế attention giúp mô hình biết lúc nào đọc chữ nào
  - Đơn giản hóa pipeline TTS

### 1.3. Thế hệ hiện đại (2018-nay)
#### Tacotron 2 (2018)
- Kết hợp Tacotron (tạo tần số) + WaveNet (vocoder: dịch tần số thành âm thanh).
- Giọng rất tự nhiên.
- Nhược điểm: biên dịch chậm.

#### FastSpeech (2019)
- Non-autoregressive → sinh nhanh, không lỗi lặp từ.
- Dùng Variance Adaptor (biên độ, cao độ, mạnh hoặc nhẹ).
- Tối ưu cho tốc độ và độ ổn định.

#### VITS (2021)
- Mô hình end-to-end: gộp acoustic model + vocoder.
- Tự xử lý từ văn bản → giọng nói, không cần nhiều bước trung gian.
- Chất lượng tốt + tốc độ nhanh.

#### Neural Codec Models (2022–nay)
- Ví dụ: VALL-E, AudioLM.
- Chuyển âm thanh thành các tokens (EnCodec, DAC).
- TTS hoạt động như LLM → dự đoán token tiếp theo.
- Zero-shot giả giọng từ mẫu 3–10s.
- Biểu cảm tốt nhưng cần GPU mạnh.


---

## 2. Kiến trúc TTS hiện đại

### 2.1. Pipeline TTS truyền thống

```
Text Input → Text Analysis → Acoustic Model → Vocoder → Audio Output
```
**Các thành phần chính:**

1. **Phân tích văn bản:**
   - Tách từ và chuẩn hóa văn bản.
   - Chuyển đổi chữ viết thành âm vị (Grapheme-to-Phoneme).
   - Dự đoán ngữ điệu.

2. **Mô hình âm học (Acoustic Model):**
   - Nhiệm vụ: Dự đoán biểu đồ phổ Mel (mel-spectrogram) từ văn bản.
   - Các mô hình tiêu biểu: Tacotron, FastSpeech, Transformer TTS.

3. **Bộ tạo tiếng (Vocoder):**
   - Nhiệm vụ: Chuyển biểu đồ phổ Mel thành sóng âm thanh nghe được.
   - Các mô hình tiêu biểu: WaveNet, WaveGlow, HiFi-GAN.

### 2.2. Kiến trúc Tacotron 2

**Cấu tạo chi tiết:**

1. **Nhúng ký tự (Character Embeddings):**
   - Chuyển đổi văn bản thành dạng vector số học (số chiều: 512).

2. **Bộ mã hóa (Encoder):**
   - Sử dụng 3 lớp tích chập (convolutional layers) kết hợp chuẩn hóa và hàm kích hoạt ReLU.
   - Dùng mạng LSTM hai chiều để nắm bắt ngữ cảnh của câu văn.

3. **Cơ chế sự chú ý (Attention Mechanism):**
   - Sử dụng loại chú ý nhạy cảm với vị trí (Location-sensitive attention) để đảm bảo mô hình không bị lạc trôi khi đọc.
   - Giúp căn chỉnh khớp giữa văn bản và biểu đồ phổ âm thanh.

4. **Bộ giải mã (Decoder):**
   - Gồm 2 lớp LSTM với ngữ cảnh từ cơ chế chú ý.
   - Dự đoán biểu đồ phổ Mel (80 dải tần).

5. **Mạng hậu xử lý (Post-net):**
   - Gồm 5 lớp tích chập.
   - Tinh chỉnh lại biểu đồ phổ đầu ra cho mượt mà hơn.

### 2.3. Kiến trúc FastSpeech 2

**Cải tiến so với Tacotron:**

1. **Feed-Forward Transformer:**
   - Sinh song song toàn bộ biểu đồ phổ cùng lúc thay vì sinh tuần tự.
   - Tốc độ cực nhanh.

2. **Bộ điều chỉnh biến thiên (Variance Adaptors):**
   - **Dự đoán độ dài:** Xác định mỗi âm vị sẽ kéo dài bao lâu.
   - **Dự đoán cao độ:** Xác định độ trầm bổng của giọng nói.
   - **Dự đoán năng lượng:** Xác định độ to nhỏ (mạnh/nhẹ) của âm thanh.

3. **Bộ điều chỉnh độ dài (Length Regulator):**
   - Mở rộng vector âm vị dựa trên dự đoán độ dài để khớp với thời gian thực của file âm thanh.

**Ưu điểm:**
- Có khả năng kiểm soát cao (tùy chỉnh được tốc độ, cao độ, âm lượng).
- Ổn định hơn, tránh được các lỗi hay gặp ở mô hình tự hồi quy.
- Phù hợp để triển khai cho sản phẩm thực tế.

### 2.4. Bộ tạo tiếng: HiFi-GAN

**Sử dụng Mạng đối kháng sinh (GAN) để tạo âm thanh:**

1. **Bộ sinh (Generator):**
   - Sử dụng kỹ thuật tích chập chuyển đổi (Transpose convolutions) để phóng to dữ liệu.
   - Kết hợp đa trường tiếp nhận (Multi-receptive field fusion) để tái tạo chi tiết âm thanh tốt hơn.

2. **Bộ phân biệt (Discriminators):**
   - Phân tích các mẫu tuần hoàn của âm thanh.
   - Phân tích âm thanh ở nhiều tỷ lệ/độ phân giải khác nhau.

**Đặc điểm:**
- Có thể chạy thời gian thực ngay trên CPU.
- Chất lượng âm thanh rất cao.
- Mô hình nhẹ và hiệu quả.

---

## 3. Các kỹ thuật nâng cao

### 3.1. Tổng hợp đa giọng nói (Multi-Speaker TTS)
- **Vector đặc trưng người nói (Speaker Embeddings):** Học các đặc điểm riêng biệt của từng giọng nói (sử dụng d-vector hoặc x-vector).
- **Mô hình có điều kiện:** Kết hợp thông tin văn bản với vector đặc trưng người nói để tạo ra giọng đúng với người được yêu cầu.
- **Zero-Shot TTS:** Các mô hình như VALL-E có thể sao chép giọng nói mới chỉ cần một đoạn mẫu ngắn mà không cần huấn luyện lại.

### 3.2. TTS biểu cảm và cảm xúc
- **Token phong cách (Style Tokens):** Học các biểu diễn ẩn về ngữ điệu và phong cách nói.
- **Bộ mã hóa tham chiếu (Reference Encoder):** Trích xuất phong cách từ một file âm thanh mẫu và áp dụng nó vào giọng nói được tạo ra.
- **Kiểm soát trực tiếp:** Gán nhãn cảm xúc cụ thể (vui, buồn, giận, bình thường) hoặc gắn thẻ ngữ điệu (nhấn mạnh, ngắt nghỉ).

### 3.3. TTS đa ngôn ngữ và xuyên ngôn ngữ
- **Thách thức:** Hệ thống âm vị khác nhau, ngữ điệu khác nhau giữa các ngôn ngữ.
- **Giải pháp:**
  - Sử dụng bảng phiên âm quốc tế (IPA) làm chuẩn chung.
  - Sử dụng các lớp nhúng ngôn ngữ (Language Embeddings) để mô hình nhận biết đang xử lý tiếng nước nào.
  - Áp dụng học chuyển tiếp (Transfer Learning): Huấn luyện trên ngôn ngữ phổ biến trước, sau đó tinh chỉnh cho ngôn ngữ ít dữ liệu.

---

## 4. Đánh giá chất lượng TTS

### 4.1. Các chỉ số khách quan (Máy đo)

| Chỉ số | Mô tả | Mục tiêu |
|--------|-------|----------|
| **MCD (Mel Cepstral Distortion)** | Đo độ sai lệch của các hệ số phổ | Càng thấp càng tốt |
| **F0 RMSE** | Sai số trung bình phương của tần số cơ bản | Càng thấp càng tốt |
| **Spectrogram Loss** | Độ sai lệch giữa biểu đồ phổ dự đoán và thực tế | Càng thấp càng tốt |

### 4.2. Các chỉ số chủ quan (Người đánh giá)

| Chỉ số | Mô tả | Thang điểm |
|--------|-------|------------|
| **MOS (Mean Opinion Score)** | Điểm đánh giá chất lượng tổng thể | 1-5 (5 là tốt nhất) |
| **CMOS** | So sánh trực tiếp giữa 2 hệ thống | -3 đến +3 |
| **WER (Word Error Rate)** | Tỷ lệ lỗi từ (đo độ rõ bằng cách dùng máy nhận dạng lại) | Càng thấp càng tốt |

---

## 5. Ứng dụng thực tế

1.  **Trợ lý ảo (Google Assistant, Siri, Alexa):** Yêu cầu độ trễ thấp, giọng nói tự nhiên và biểu cảm.
2.  **Sách nói và Podcast:** Yêu cầu khả năng đọc văn bản dài, giọng đọc nhất quán, hỗ trợ nhiều vai/giọng khác nhau.
3.  **Hỗ trợ tiếp cận (Màn hình đọc, Dẫn đường):** Yêu cầu phát âm rõ ràng, hỗ trợ đa ngôn ngữ, chạy nhẹ nhàng trên thiết bị.
4.  **Sáng tạo nội dung (Lồng tiếng video, Game):** Yêu cầu kiểm soát được cảm xúc, giọng nhân vật, tạo giọng theo thời gian thực.
5.  **Y tế:** Thiết bị hỗ trợ giao tiếp cho người khuyết tật, ngân hàng giọng nói (lưu giữ giọng nói cho bệnh nhân trước khi mất khả năng nói).

---

## 6. Thách thức và hướng phát triển

### 6.1. Thách thức hiện tại
- **Ngữ điệu và Biểu cảm:** Khó kiểm soát các chi tiết nhỏ trong ngữ điệu, thiếu tự nhiên khi đọc văn bản dài.
- **Dữ liệu:** Cần lượng lớn dữ liệu chất lượng cao (phòng thu). Chi phí thu thập giọng hiếm rất đắt đỏ. Vấn đề bảo mật dữ liệu giọng nói.
- **Chi phí tính toán:** Các bộ tạo tiếng (vocoder) chất lượng cao vẫn khá nặng, khó đưa lên các thiết bị nhỏ (điện thoại, IoT).
- **Đánh giá:** Đánh giá bằng người thật (MOS) tốn kém, trong khi đánh giá bằng máy chưa phản ánh đúng cảm nhận của con người.

### 6.2. Hướng nghiên cứu tương lai
- **Zero-Shot và Few-Shot:** Sao chép giọng nói từ dữ liệu cực ít. Chuyển đổi giọng nói xuyên ngôn ngữ.
- **Mô hình thống nhất:** Một mô hình làm được nhiều việc (vừa nghe, vừa nói, vừa chuyển đổi giọng).
- **Khả năng kiểm soát:** Tách biệt nội dung và phong cách nói để điều chỉnh dễ dàng hơn.
- **Hiệu năng:** Làm nhẹ mô hình để chạy trên di động.
- **Đạo đức và An toàn:** Phát hiện Deepfake, đóng dấu bản quyền giọng nói (watermarking), xác thực người nói.

---

## 7. Công cụ và Framework

### 7.1. Mã nguồn mở
- **Coqui TTS:** Hỗ trợ nhiều mô hình (Tacotron, VITS...), viết bằng Python.
- **ESPnet:** Toolkit mạnh về xử lý tiếng nói từ CMU.
- **Mozilla TTS:** Dự án TTS của Mozilla.
- **PaddleSpeech:** Framework của Baidu.

### 7.2. Dịch vụ đám mây (Cloud APIs)
- **Google Cloud TTS, Amazon Polly, Microsoft Azure TTS:** Cung cấp hàng trăm giọng đọc, hỗ trợ đa ngôn ngữ, tính phí theo ký tự.
- **ElevenLabs:** Nổi bật với khả năng sao chép giọng nói (voice cloning) và biểu cảm cực tốt.

---

## 8. So sánh các phương pháp TTS

### 8.1. Tự hồi quy (Autoregressive) vs Phi tự hồi quy (Non-Autoregressive)

| Tiêu chí | Tự hồi quy (Tacotron) | Phi tự hồi quy (FastSpeech) |
| :--- | :--- | :--- |
| **Tốc độ** | Chậm (do xử lý tuần tự từng bước) | Nhanh (xử lý song song) |
| **Độ ổn định** | Hay gặp lỗi lặp từ, bỏ từ hoặc mất sự chú ý | Ổn định, bền vững, không lỗi sự chú ý |
| **Chất lượng** | Ngữ điệu rất tự nhiên | Có thể hơi máy móc nếu không tinh chỉnh kỹ |
| **Huấn luyện** | Đơn giản | Phức tạp hơn (cần bộ dự đoán thời lượng) |
| **Khả năng kiểm soát** | Hạn chế | Dễ dàng điều chỉnh thời lượng, cao độ |

### 8.2. Vocoder dùng GAN vs Vocoder dùng Flow

| Tiêu chí | GAN (HiFi-GAN) | Flow (WaveGlow) |
| :--- | :--- | :--- |
| **Huấn luyện** | Không ổn định, cần tinh chỉnh nhiều | Ổn định hơn |
| **Chất lượng** | Chất lượng cao | Chất lượng cao |
| **Tốc độ** | Rất nhanh | Chậm hơn |
| **Bộ nhớ** | Hiệu quả, tốn ít tài nguyên | Tốn nhiều bộ nhớ |
| **Nhiễu/Sạn** | Thỉnh thoảng có lỗi nhỏ (glitch) | Âm thanh sạch hơn |

---

## 9. Kết luận

Công nghệ chuyển đổi văn bản thành giọng nói (Text-to-Speech) đã có những bước tiến vượt bậc trong thập kỷ qua nhờ sự bùng nổ của học sâu (Deep Learning). Các mô hình hiện đại như Tacotron 2, FastSpeech 2 và VITS đã đạt được chất lượng gần như tự nhiên, xóa nhòa ranh giới giữa giọng máy và người, mở ra nhiều ứng dụng thực tế.

**Các điểm cốt lõi cần nhớ:**

1.  **Quy trình chuẩn (Pipeline):** Phân tích văn bản → Mô hình âm học → Bộ tạo tiếng (Vocoder).
2.  **Mô hình âm học:** Sự chuyển dịch từ Tacotron 2 (tự hồi quy, chậm) sang FastSpeech 2 (phi tự hồi quy, nhanh).
3.  **Bộ tạo tiếng (Vocoder):** HiFi-GAN và WaveGlow là chuẩn mực cho âm thanh chất lượng cao.
4.  **Chủ đề nâng cao:** Tổng hợp đa giọng nói, giọng nói biểu cảm cảm xúc, và kỹ thuật Zero-shot (sao chép giọng nhanh).
5.  **Sự đánh đổi:** Luôn phải cân nhắc giữa Chất lượng vs Tốc độ vs Khả năng kiểm soát.

**Xu hướng tương lai:**
- **Mô hình nền tảng (Foundation models):** Hướng tới việc tạo sinh tiếng nói tổng quát và đa năng.
- **Sao chép giọng (Zero-shot voice cloning):** Giả lập giọng nói bất kỳ chỉ với mẫu cực ngắn.
- **Thời gian thực:** Tối ưu hóa để chạy mượt mà ngay trên thiết bị cá nhân (on-device).
- **Đạo đức AI:** Tập trung vào phát hiện Deepfake và bảo vệ bản quyền giọng nói.

---

## Tài liệu tham khảo

1. **WaveNet:** van den Oord et al., "WaveNet: A Generative Model for Raw Audio" (2016)
2. **Tacotron:** Wang et al., "Tacotron: Towards End-to-End Speech Synthesis" (2017)
3. **Tacotron 2:** Shen et al., "Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions" (2018)
4. **FastSpeech:** Ren et al., "FastSpeech: Fast, Robust and Controllable Text to Speech" (2019)
5. **FastSpeech 2:** Ren et al., "FastSpeech 2: Fast and High-Quality End-to-End Text to Speech" (2020)
6. **HiFi-GAN:** Kong et al., "HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis" (2020)
7. **VITS:** Kim et al., "Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech" (2021)
8. **VALL-E:** Wang et al., "Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers" (2023)

**Nguồn:**
- [Coqui TTS Documentation](https://github.com/coqui-ai/TTS)
- [ESPnet Speech Processing Toolkit](https://github.com/espnet/espnet)
- [Hugging Face TTS Models](https://huggingface.co/models?pipeline_tag=text-to-speech)
- [Papers with Code - TTS](https://paperswithcode.com/task/text-to-speech-synthesis)
