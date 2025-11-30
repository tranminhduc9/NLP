# Tổng Quan Về Các Dataset Trong Dự Án NLP

Tài liệu này mô tả chi tiết cấu trúc, kiểu dữ liệu và mục đích sử dụng của tất cả các dataset trong dự án.

---

## 1. Sentiments Dataset

**Đường dẫn:** `data/sentiments.csv`

### Thông tin chung
- **Số lượng mẫu:** 5,793 dòng (không tính header)
- **Định dạng:** CSV
- **Mục đích:** Phân tích cảm xúc (Sentiment Analysis) - Lab 5 Part 2

### Cấu trúc dữ liệu

| Cột | Kiểu dữ liệu | Mô tả | Ví dụ |
|-----|--------------|-------|-------|
| `text` | String | Văn bản đầu vào cần phân tích cảm xúc | "Kickers on my watchlist KXAN..." |
| `sentiment` | Integer | Nhãn cảm xúc: `1` (tích cực), `-1` (tiêu cực) | `1` |

### Ví dụ mẫu
```csv
text,sentiment
"Kickers on my watchlist KXAN ...",1
"$MAR Remains on the bearish trend...",1
"@Mila2021 $TWTR near the close...",1
```

### Đặc điểm
- Dataset cân bằng với 2 lớp (positive/negative)
- Văn bản ngắn, chủ yếu từ Twitter/social media
- Sử dụng trong các bài lab về phân loại văn bản với TF-IDF, Word2Vec

---

## 2. HWU64 Intent Classification Dataset

**Đường dẫn:** `data/hwu/`

### Thông tin chung
- **Tổng số mẫu:** 11,110 câu
  - Training set: 8,955 mẫu (`train.csv`)
  - Validation set: 1,077 mẫu (`val.csv`)
  - Test set: 1,077 mẫu (`test.csv`)
- **Định dạng:** CSV
- **Số lượng lớp:** 64 intent categories
- **Mục đích:** Phân loại Intent (Intent Classification) - Lab 5 Part 2

### Cấu trúc dữ liệu

| Cột | Kiểu dữ liệu | Mô tả | Ví dụ |
|-----|--------------|-------|-------|
| `text` | String | Câu hỏi/câu lệnh của người dùng | "what alarms do i have set right now" |
| `category` | String | Intent label - 1 trong 64 categories | "alarm_query" |

### Danh sách 64 Intent Categories

File `categories.json` chứa danh sách đầy đủ các categories:

```json
[
  "alarm_query", "alarm_remove", "alarm_set", "audio_volume_down",
  "audio_volume_mute", "audio_volume_other", "audio_volume_up",
  "calendar_query", "calendar_remove", "calendar_set", 
  "cooking_query", "cooking_recipe", "datetime_convert", "datetime_query",
  "email_addcontact", "email_query", "email_querycontact", "email_sendemail",
  "general_affirm", "general_commandstop", "general_confirm", "general_dontcare",
  "general_explain", "general_greet", "general_joke", "general_negate",
  "general_praise", "general_quirky", "general_repeat",
  "iot_cleaning", "iot_coffee", "iot_hue_lightchange", "iot_hue_lightdim",
  "iot_hue_lightoff", "iot_hue_lighton", "iot_hue_lightup", "iot_wemo_off",
  "iot_wemo_on", "lists_createoradd", "lists_query", "lists_remove",
  "music_dislikeness", "music_likeness", "music_query", "music_settings",
  "news_query", "play_audiobook", "play_game", "play_music", "play_podcasts",
  "play_radio", "qa_currency", "qa_definition", "qa_factoid", "qa_maths",
  "qa_stock", "recommendation_events", "recommendation_locations",
  "recommendation_movies", "social_post", "social_query",
  "takeaway_order", "takeaway_query", "transport_query", "transport_taxi",
  "transport_ticket", "transport_traffic", "weather_query"
]
```

### Ví dụ mẫu
```csv
text,category
"what alarms do i have set right now",alarm_query
"olly please stop the alarm",alarm_remove
"set an alarm for five am",alarm_set
"tell me the weather today",weather_query
"play my favorite song",play_music
```

### Đặc điểm
- Dataset cho hệ thống trợ lý ảo (Virtual Assistant)
- Bao gồm các domain: alarm, calendar, IoT, music, weather, email, etc.
- Văn bản ngắn (1-2 câu), conversational style
- Sử dụng trong các bài lab về text classification với neural networks

---

## 3. Universal Dependencies English Web Treebank (UD_English-EWT)

**Đường dẫn:** `data/UD_English-EWT/`

### Thông tin chung
- **Phiên bản:** v2.13 (2023-11-15)
- **Tổng số mẫu:** 16,622 câu, 254,820 từ
- **Định dạng:** CoNLL-U format
- **Mục đích:** Part-of-Speech Tagging và Dependency Parsing - Lab 5 Part 3
- **Nguồn:** LDC2012T13 - English Web Treebank
- **License:** CC BY-SA 4.0

### Phân chia dữ liệu

| File | Số dòng | Số câu (ước tính) | Mô tả |
|------|---------|-------------------|-------|
| `en_ewt-ud-train.conllu` | 247,862 | ~12,543 | Training set |
| `en_ewt-ud-dev.conllu` | 30,581 | ~2,002 | Development/Validation set |
| `en_ewt-ud-test.conllu` | 30,774 | ~2,077 | Test set |

### Cấu trúc CoNLL-U Format

Mỗi từ được annotation với 10 cột:

| Cột | Tên | Kiểu | Mô tả | Ví dụ |
|-----|-----|------|-------|-------|
| 1 | ID | Integer | Chỉ số từ trong câu (bắt đầu từ 1) | `1` |
| 2 | FORM | String | Surface form của từ | `American` |
| 3 | LEMMA | String | Lemma (dạng gốc) của từ | `American` |
| 4 | UPOS | String | Universal POS tag (17 tags) | `ADJ` |
| 5 | XPOS | String | Language-specific POS tag (Penn Treebank) | `JJ` |
| 6 | FEATS | String | Morphological features | `Degree=Pos` |
| 7 | HEAD | Integer | ID của từ cha trong cây dependency | `6` |
| 8 | DEPREL | String | Dependency relation với từ cha | `amod` |
| 9 | DEPS | String | Enhanced dependencies | `6:amod` |
| 10 | MISC | String | Thông tin bổ sung | `SpaceAfter=No` |

### Universal POS Tags (17 tags)

```
ADJ    - Adjective
ADP    - Adposition
ADV    - Adverb
AUX    - Auxiliary
CCONJ  - Coordinating conjunction
DET    - Determiner
INTJ   - Interjection
NOUN   - Noun
NUM    - Numeral
PART   - Particle
PRON   - Pronoun
PROPN  - Proper noun
PUNCT  - Punctuation
SCONJ  - Subordinating conjunction
SYM    - Symbol
VERB   - Verb
X      - Other
```

### Ví dụ mẫu

```conllu
# sent_id = weblog-juancole.com_juancole_20051126063000_ENG_20051126_063000-0001
# text = American forces killed Shaikh Abdullah.
1	American	American	ADJ	JJ	Degree=Pos	2	amod	2:amod	_
2	forces	force	NOUN	NNS	Number=Plur	3	nsubj	3:nsubj	_
3	killed	kill	VERB	VBD	Mood=Ind|Tense=Past	0	root	0:root	_
4	Shaikh	Shaikh	PROPN	NNP	Number=Sing	3	obj	3:obj	_
5	Abdullah	Abdullah	PROPN	NNP	Number=Sing	4	flat	4:flat	SpaceAfter=No
6	.	.	PUNCT	.	_	3	punct	3:punct	_
```

### Metadata trong file

- **Sentence ID:** `# sent_id = <id>`
- **Original text:** `# text = <sentence>`
- **Document ID:** `# newdoc id = <id>`
- **Paragraph ID:** `# newpar id = <id>`

### Đặc điểm
- Dữ liệu từ 5 thể loại: weblogs, newsgroups, emails, reviews, Yahoo! answers
- Annotation thủ công (hand-corrected)
- Bao gồm cả basic và enhanced dependencies
- Được sử dụng rộng rãi trong nghiên cứu NLP
- Compatible với Universal Dependencies standard

---

## 4. C4 (Colossal Clean Crawled Corpus) Dataset

**Đường dẫn:** `data/c4-train.00000-of-01024-30K.json`

### Thông tin chung
- **Kích thước file:** >50 MB (không thể đọc trực tiếp bằng công cụ)
- **Định dạng:** JSON (mỗi dòng là 1 JSON object)
- **Số lượng mẫu:** ~30,000 documents (dựa trên tên file)
- **Mục đích:** Training Word2Vec embeddings - Lab 4

### Cấu trúc dữ liệu (dự kiến)

```json
{
  "text": "Lorem ipsum dolor sit amet, consectetur adipiscing elit...",
  "timestamp": "2019-01-01T00:00:00Z",
  "url": "https://example.com/page"
}
```

| Trường | Kiểu dữ liệu | Mô tả |
|--------|--------------|-------|
| `text` | String | Văn bản đã được làm sạch từ web |
| `timestamp` | String (ISO 8601) | Thời gian crawl dữ liệu |
| `url` | String | URL nguồn của văn bản |

### Đặc điểm
- Dữ liệu đã được làm sạch (clean) từ Common Crawl
- Chất lượng cao, loại bỏ nội dung spam/low-quality
- Sử dụng để train word embeddings (Word2Vec) trong Lab 4
- File này là 1 phần của bộ dataset lớn hơn (00000-of-01024)

---

## Tổng Kết

| Dataset | Số mẫu | Task | Lab |
|---------|--------|------|-----|
| **sentiments.csv** | 5,793 | Sentiment Analysis (Binary) | Lab 5 Part 2 |
| **hwu/** | 11,110 | Intent Classification (64 classes) | Lab 5 Part 2 |
| **UD_English-EWT/** | 16,622 câu | POS Tagging & Dependency Parsing | Lab 5 Part 3 |
| **c4-train.json** | ~30,000 | Word Embedding Training | Lab 4 |

### Phân phối Dataset theo Lab

- **Lab 1-2:** Regex Tokenizer, Simple Tokenizer (không cần dataset cụ thể)
- **Lab 3:** Count Vectorizer, TF-IDF (sử dụng sentiments.csv)
- **Lab 4:** Word2Vec, GloVe training (sử dụng c4-train.json và UD_English-EWT)
- **Lab 5 Part 1:** Text Classification với Word2Vec (sentiments.csv, hwu/)
- **Lab 5 Part 2:** Sentiment Analysis với PySpark (sentiments.csv)
- **Lab 5 Part 3:** POS Tagging với RNN (UD_English-EWT/)
- **Lab 5 Part 4:** Named Entity Recognition với BiLSTM (CoNLL-2003 từ Hugging Face)
- **Lab 6:** Transformers (BERT, GPT-2, Sentence Embeddings)
