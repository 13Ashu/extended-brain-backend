# Extended Brain - Complete Backend System

WhatsApp-powered AI knowledge base with Cerebras LLM and PostgreSQL (Neon).

## 🎯 Features

- ✅ **WhatsApp Integration**: Send text, images, audio, PDFs via WhatsApp
- ✅ **AI-Powered Categorization**: Automatic organization using Cerebras LLM
- ✅ **Semantic Search**: Find anything in your knowledge base
- ✅ **Multi-format Support**: Text, images, audio, documents
- ✅ **Dynamic Categories**: Create, edit, delete, merge categories
- ✅ **Smart Tagging**: Auto-tag messages with relevant keywords
- ✅ **Entity Extraction**: Extract people, dates, locations automatically

## 📁 Project Structure

```
extended-brain-backend/
├── main.py                          # FastAPI application
├── database.py                      # SQLAlchemy models & config
├── models.py                        # Model exports
├── cerebras_client.py               # Cerebras AI integration
├── whatsapp.py                      # WhatsApp Business API
├── services/
│   ├── message_processor.py        # Process incoming messages
│   ├── search_service.py           # Search functionality
│   ├── category_manager.py         # Category management
│   └── document_processor.py       # PDF/DOCX extraction
├── requirements.txt                 # Python dependencies
├── .env.example                     # Environment variables template
├── Procfile                         # Railway deployment
├── railway.json                     # Railway config
└── README.md                        # This file
```

## 🚀 Quick Setup

### Step 1: Create Project Folder

```bash
mkdir extended-brain-backend
cd extended-brain-backend
```

### Step 2: Download All Files

Download all the numbered files and rename them:

- `1_main.py` → `main.py`
- `2_database.py` → `database.py`
- `3_models.py` → `models.py`
- `4_cerebras_client.py` → `cerebras_client.py`
- `5_whatsapp.py` → `whatsapp.py`
- `6_services_message_processor.py` → `services/message_processor.py`
- `7_services_search_service.py` → `services/search_service.py`
- `8_services_category_manager.py` → `services/category_manager.py`
- `9_services_document_processor.py` → `services/document_processor.py`
- `10_requirements.txt` → `requirements.txt`
- `11_env_example.txt` → `.env.example`
- `12_Procfile.txt` → `Procfile`
- `13_railway.json` → `railway.json`

**IMPORTANT**: Create a `services` folder for files 6-9!

```bash
mkdir services
# Move the service files into this folder
```

### Step 3: Install Dependencies

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate  # Mac/Linux
# OR
venv\Scripts\activate     # Windows

# Install packages
pip install -r requirements.txt
```

### Step 4: Setup Environment Variables

```bash
# Copy example file
cp .env.example .env

# Edit .env and add your credentials
nano .env  # or use any text editor
```

You need:
- **DATABASE_URL** from Neon.tech
- **CEREBRAS_API_KEY** from Cerebras Cloud
- **WHATSAPP credentials** from Meta Business

## 🗄️ Database Setup (Neon)

1. Sign up at https://neon.tech
2. Create a new project
3. Copy the connection string
4. Add to `.env` file

```env
DATABASE_URL=postgresql://user:pass@ep-xxx.us-east-2.aws.neon.tech/extendedbrain?sslmode=require
```

## 🧠 Cerebras API Setup

1. Sign up at https://cloud.cerebras.ai
2. Get your API key
3. Add to `.env` file

```env
CEREBRAS_API_KEY=csk-your-api-key-here
```

## 📱 WhatsApp Setup

1. Go to https://developers.facebook.com
2. Create a Business app
3. Add WhatsApp product
4. Get access token and phone number ID
5. Add to `.env` file

```env
WHATSAPP_ACCESS_TOKEN=your_token
WHATSAPP_PHONE_NUMBER_ID=your_phone_id
WHATSAPP_VERIFY_TOKEN=create_any_random_string
```

## 🧪 Test Locally

```bash
# Start the server
python main.py

# Server runs at http://localhost:8000
# API docs at http://localhost:8000/docs
```

## 🚀 Deploy to Railway

1. Push code to GitHub
2. Go to https://railway.app
3. Create new project from GitHub repo
4. Add environment variables
5. Deploy!

**Your URL**: `https://your-app.railway.app`

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info |
| `/health` | GET | Health check |
| `/webhook` | GET/POST | WhatsApp webhook |
| `/api/messages/capture` | POST | Capture message |
| `/api/search` | POST | Search messages |
| `/api/categories/manage` | POST | Manage categories |
| `/api/users/register` | POST | Register user |

## 💰 Cost Estimate

- **Neon**: Free tier (0.5 GB)
- **Railway**: $5/month
- **Cerebras**: ~$0.60 per 1M tokens
- **WhatsApp**: Free (first 1000 conversations/month)

**Total**: ~$5-10/month

## 📝 License

MIT
