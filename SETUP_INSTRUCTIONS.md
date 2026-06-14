# 🚀 SETUP INSTRUCTIONS - Extended Brain Backend

## 📥 Step-by-Step File Setup

### 1. Create Main Project Folder

```bash
mkdir extended-brain-backend
cd extended-brain-backend
```

### 2. Download and Rename Files

Download all files from Claude and organize them:

**Root Files:**
```
1_main.py           → rename to → main.py
2_database.py       → rename to → database.py
3_models.py         → rename to → models.py
4_cerebras_client.py → rename to → cerebras_client.py
5_whatsapp.py       → rename to → whatsapp.py
10_requirements.txt → rename to → requirements.txt
11_env_example.txt  → rename to → .env.example
12_Procfile.txt     → rename to → Procfile
13_railway.json     → rename to → railway.json
14_README.md        → rename to → README.md
```

**Services Folder (CREATE THIS!):**
```bash
mkdir services
```

Then move these files into the `services/` folder:
```
6_services_message_processor.py  → rename to → services/message_processor.py
7_services_search_service.py     → rename to → services/search_service.py
8_services_category_manager.py   → rename to → services/category_manager.py
9_services_document_processor.py → rename to → services/document_processor.py
```

### 3. Final Folder Structure

Your folder should look like this:

```
extended-brain-backend/
├── main.py
├── database.py
├── models.py
├── cerebras_client.py
├── whatsapp.py
├── services/
│   ├── message_processor.py
│   ├── search_service.py
│   ├── category_manager.py
│   └── document_processor.py
├── requirements.txt
├── .env.example
├── Procfile
├── railway.json
└── README.md
```

### 4. Create Virtual Environment

```bash
# On Windows:
python -m venv venv
venv\Scripts\activate

# On Mac/Linux:
python3 -m venv venv
source venv/bin/activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Setup Environment Variables

```bash
# Copy the example file
cp .env.example .env

# Edit .env with your credentials
# You can use notepad, nano, vim, or any text editor
notepad .env  # Windows
nano .env     # Mac/Linux
```

Add your credentials:
```env
DATABASE_URL=postgresql://YOUR_RAILWAY_POSTGRES_URL_HERE
CEREBRAS_API_KEY=YOUR_CEREBRAS_KEY_HERE
WHATSAPP_ACCESS_TOKEN=YOUR_WHATSAPP_TOKEN
WHATSAPP_PHONE_NUMBER_ID=YOUR_PHONE_ID
WHATSAPP_VERIFY_TOKEN=any_random_string_you_create
```

### 7. Get Your API Keys

**Railway PostgreSQL:**
1. In your Railway project: + New → Database → PostgreSQL
2. Railway provisions it and sets `DATABASE_URL` automatically
3. For local dev, copy it from the Postgres service → Variables

**Cerebras API:**
1. Go to https://cloud.cerebras.ai
2. Sign up
3. Get API key from dashboard

**WhatsApp Business:**
1. Go to https://developers.facebook.com
2. Create Business app
3. Add WhatsApp product
4. Get access token and phone number ID

### 8. Test Locally

```bash
python main.py
```

Visit:
- API: http://localhost:8000
- Docs: http://localhost:8000/docs
- Health: http://localhost:8000/health

### 9. Deploy to Railway

1. Push code to GitHub:
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin YOUR_GITHUB_REPO
git push -u origin main
```

2. Go to https://railway.app
3. Click "New Project"
4. Select your GitHub repo
5. Add environment variables (same as in .env)
6. Deploy!

## ✅ Checklist

Before running:
- [ ] All files downloaded and renamed correctly
- [ ] `services/` folder created with 4 files inside
- [ ] Virtual environment created and activated
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] `.env` file created with your credentials
- [ ] Railway PostgreSQL `DATABASE_URL` added to `.env`
- [ ] Cerebras API key added to `.env`
- [ ] WhatsApp credentials added to `.env`

## 🆘 Common Issues

**Issue**: `ModuleNotFoundError: No module named 'services'`
**Fix**: Make sure you created the `services` folder and put files 6-9 inside it

**Issue**: `Can't connect to database`
**Fix**: Check your DATABASE_URL in .env file

**Issue**: `Cerebras API error`
**Fix**: Verify your CEREBRAS_API_KEY is correct

**Issue**: Import errors
**Fix**: Make sure all files are named correctly (no numbers in filenames!)

## 📞 Need Help?

Check:
1. README.md for detailed docs
2. http://localhost:8000/docs for API documentation
3. Railway logs for deployment errors

## 🎉 You're Done!

Once everything is set up, your Extended Minds backend is ready to serve the iOS app!

Send a message to your WhatsApp Business number and watch it automatically categorize, tag, and store your information! 🚀
