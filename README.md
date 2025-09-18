# ECG Abnormality Classification API - Heroku Deployment

## 🚀 Quick Start

This directory contains all the files needed to deploy your ECG classification API to Heroku.

### Prerequisites
- [Heroku CLI](https://devcenter.heroku.com/articles/heroku-cli) installed
- Git installed
- A Heroku account (free tier available)

### Files in this deployment package:
- `ecg_api_server.py` - Production-ready FastAPI server
- `requirements.txt` - Python dependencies
- `Procfile` - Heroku process definition
- `runtime.txt` - Python version specification
- `.gitignore` - Git ignore rules
- `test_api.py` - API testing client
- `web_interface.html` - Web-based testing interface

## 📦 Deployment Steps

### 1. Copy Your Model File
Copy your trained model file `ecg_abnormality_classifier_lightgbm.pkl` to this deployment directory:
```bash
cp ../ecg_abnormality_classifier_lightgbm.pkl ./
```

### 2. Initialize Git Repository
```bash
git init
git add .
git commit -m "Initial commit - ECG classification API"
```

### 3. Login to Heroku
```bash
heroku login
```

### 4. Create Heroku App
```bash
# Replace 'your-ecg-api' with your preferred app name (must be unique)
heroku create your-ecg-api
```

### 5. Deploy to Heroku
```bash
git push heroku main
```
*Note: If your default branch is 'master', use `git push heroku master`*

### 6. Scale the Application
```bash
heroku ps:scale web=1
```

### 7. Open Your API
```bash
heroku open
```

## 🔧 Testing Your Deployed API

### Method 1: Using the Web Interface
1. Open `web_interface.html` in your browser
2. Update the API URL to your Heroku app URL
3. Use the form to test predictions

### Method 2: Using Python Client
```python
python test_api.py
# Edit the script to use your actual Heroku URL
```

### Method 3: Direct API Calls
Your API will be available at: `https://your-app-name.herokuapp.com`

**Available endpoints:**
- `GET /` - Root endpoint
- `GET /health` - Health check
- `GET /docs` - Interactive API documentation
- `GET /model/info` - Model information
- `POST /predict` - Single prediction
- `POST /predict/batch` - Batch predictions
- `POST /predict/csv` - CSV file predictions

## 📊 API Features

### 🔍 Single Prediction
```bash
curl -X POST "https://your-app-name.herokuapp.com/predict?threshold=0.5" \
     -H "Content-Type: application/json" \
     -d '{"Age": 65, "Gender": "M", "ECG_QTC_INTERVAL": 420.5, "ECG_QRS_DURATION": 98.2}'
```

### 📋 Batch Prediction
```bash
curl -X POST "https://your-app-name.herokuapp.com/predict/batch" \
     -H "Content-Type: application/json" \
     -d '{
       "data": [
         {"Age": 45, "Gender": "F", "ECG_QTC_INTERVAL": 400, "ECG_QRS_DURATION": 85},
         {"Age": 70, "Gender": "M", "ECG_QTC_INTERVAL": 450, "ECG_QRS_DURATION": 110}
       ],
       "threshold": 0.5
     }'
```

## 🛠️ Troubleshooting

### Common Issues:

1. **Build Fails Due to Large Model File**
   - Heroku has a 500MB slug size limit
   - Consider model compression if needed

2. **Memory Errors (R14)**
   - Free tier has 512MB memory limit
   - Upgrade to paid dyno: `heroku ps:scale web=1 --size=standard-1x`

3. **App Crashes on Startup (H10)**
   - Check logs: `heroku logs --tail`
   - Ensure model file is accessible

4. **Cold Start Issues**
   - Free dynos sleep after 30 minutes of inactivity
   - Consider paid dynos for production use

### Useful Commands:
```bash
# View logs
heroku logs --tail

# Check app status
heroku ps

# Restart app
heroku restart

# Set environment variables
heroku config:set MODEL_PATH=ecg_abnormality_classifier_lightgbm.pkl
```

## 📈 Performance Expectations

- **Model Accuracy**: 97.48% ROC-AUC
- **Response Time**: < 200ms for single predictions
- **Batch Processing**: Efficient handling of multiple samples
- **Uptime**: 99.9% on paid tiers

## 🔒 Security Notes

- API includes CORS middleware for cross-origin requests
- Input validation using Pydantic models
- Error handling and logging for debugging
- No sensitive data stored in the application

## 📱 Production Considerations

1. **Scaling**: Use appropriate dyno types for your workload
2. **Monitoring**: Set up logging and monitoring
3. **Backup**: Keep model files backed up
4. **Updates**: Use CI/CD for automated deployments
5. **Security**: Consider API authentication for production use

---

**Need help?** Check the [Heroku documentation](https://devcenter.heroku.com/) or review the logs with `heroku logs --tail`