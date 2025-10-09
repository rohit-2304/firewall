# Updated Flask WAF Application with BERT Model Integration

from flask import Flask, request, jsonify
import logging
from datetime import datetime, timezone
import json
import hashlib
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling
import re
from urllib.parse import unquote
import os

app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# ---------- BERT Model Configuration ----------
MODEL_DIR = os.environ.get('MODEL_PATH', 'bert-base-uncased')  # Default to base model if custom not found
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = 128
NUM_RANDOM_MASKS = 4  # Number of masking iterations for PLL calculation
MLM_PROB = 0.15

# Global model variables
tokenizer = None
model = None
data_collator = None

def load_bert_model():
    """Load BERT model and tokenizer for WAF inference"""
    global tokenizer, model, data_collator
    
    try:
        logger.info(f"Loading BERT model from {MODEL_DIR}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model = AutoModelForMaskedLM.from_pretrained(MODEL_DIR)
        model.to(DEVICE)
        model.eval()
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, 
            mlm=True, 
            mlm_probability=MLM_PROB
        )
        
        logger.info(f"BERT model loaded successfully on {DEVICE}")
        
    except Exception as e:
        logger.error(f"Failed to load custom model from {MODEL_DIR}, falling back to bert-base-uncased: {e}")
        try:
            tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            model = AutoModelForMaskedLM.from_pretrained('bert-base-uncased')
            model.to(DEVICE)
            model.eval()
            
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer, 
                mlm=True, 
                mlm_probability=MLM_PROB
            )
            
            logger.info("Fallback BERT model loaded successfully")
            
        except Exception as fallback_e:
            logger.error(f"Failed to load fallback model: {fallback_e}")
            raise

def normalize_url(url):
    """Normalize URL by decoding and cleaning"""
    try:
        # URL decode
        decoded = unquote(url)
        # Remove extra whitespace and normalize
        normalized = re.sub(r'\s+', ' ', decoded.strip())
        return normalized
    except:
        return url

def extract_request_text(features):
    """Convert request features to BERT-compatible text format"""
    
    # Extract components
    method = features.get('method', 'GET')
    uri = normalize_url(features.get('uri', '/'))
    user_agent = features.get('user_agent', '')
    body = features.get('body', '')
    
    # Determine user agent category
    ua_category = 'Unknown'
    if 'Chrome' in user_agent:
        ua_category = 'Chrome'
    elif 'Firefox' in user_agent:
        ua_category = 'Firefox'
    elif 'Safari' in user_agent:
        ua_category = 'Safari'
    elif 'curl' in user_agent or 'wget' in user_agent:
        ua_category = 'Script'
    elif 'PowerShell' in user_agent:
        ua_category = 'PowerShell'
    
    # Get current time info
    now = datetime.now(timezone.utc)
    hour = now.hour
    dow = now.strftime('%a')
    
    # Determine size category based on URI and body length
    total_length = len(uri) + len(body)
    if total_length < 50:
        size_category = 'small'
    elif total_length < 200:
        size_category = 'medium'
    else:
        size_category = 'large'
    
    # Extract parameters info
    params = 'none'
    if '?' in uri and '=' in uri:
        params = 'params'
    
    # Create BERT-compatible text (similar to training format)
    bert_text = f"{method} path={uri} params={params} status=200 size={size_category} ref_path=/ ua={ua_category} hour={hour:02d} dow={dow}"
    
    # Include body content if present
    if body:
        bert_text += f" body={body[:100]}"  # Limit body to 100 chars
    
    return bert_text

def calculate_pll_score(text, num_masks=NUM_RANDOM_MASKS):
    """Calculate Pseudo-Log-Likelihood score for anomaly detection"""
    if not tokenizer or not model:
        logger.error("Model not loaded")
        return 0.5
    
    try:
        # Tokenize the text
        inputs = tokenizer(text, return_tensors="pt", max_length=MAX_LEN, 
                          truncation=True, padding=True)
        
        input_ids = inputs["input_ids"].to(DEVICE)
        attention_mask = inputs["attention_mask"].to(DEVICE)
        
        total_log_prob = 0.0
        num_predictions = 0
        
        with torch.no_grad():
            for _ in range(num_masks):
                # Create a copy for masking
                masked_input_ids = input_ids.clone()
                
                # Get valid positions to mask (exclude special tokens)
                valid_positions = []
                for i in range(1, input_ids.size(1) - 1):  # Skip [CLS] and [SEP]
                    if attention_mask[0, i] == 1:
                        valid_positions.append(i)
                
                if not valid_positions:
                    continue
                
                # Randomly select positions to mask
                num_to_mask = max(1, int(len(valid_positions) * MLM_PROB))
                mask_positions = np.random.choice(valid_positions, size=num_to_mask, replace=False)
                
                # Store original tokens
                original_tokens = input_ids[0, mask_positions].clone()
                
                # Apply masking
                masked_input_ids[0, mask_positions] = tokenizer.mask_token_id
                
                # Get model predictions
                outputs = model(input_ids=masked_input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                
                # Calculate log probabilities for masked tokens
                for i, pos in enumerate(mask_positions):
                    probs = torch.softmax(logits[0, pos], dim=-1)
                    original_token_id = original_tokens[i].item()
                    token_prob = probs[original_token_id].item()
                    
                    if token_prob > 0:
                        total_log_prob += np.log(token_prob)
                        num_predictions += 1
        
        # Calculate average log probability
        if num_predictions > 0:
            avg_log_prob = total_log_prob / num_predictions
            # Convert to a score between 0 and 1 (lower PLL = higher anomaly score)
            # Using sigmoid to normalize
            anomaly_score = 1 / (1 + np.exp(avg_log_prob + 5))  # +5 for better scaling
            return float(anomaly_score)
        
        return 0.5  # Neutral score if no predictions
        
    except Exception as e:
        logger.error(f"Error calculating PLL score: {e}")
        return 0.5

def extract_features(req=None):
    """Extract relevant features from the HTTP request"""
    if req is None:
        req = request
    
    # Generate request ID if not provided
    request_id = req.headers.get('X-Request-ID')
    if not request_id:
        request_id = hashlib.md5(
            f"{datetime.utcnow().isoformat()}{req.remote_addr}{req.path}".encode()
        ).hexdigest()
    
    features = {
        'method': req.method,
        'uri': req.full_path if req.query_string else req.path,
        'client_ip': req.headers.get('X-Forwarded-For', req.remote_addr),
        'user_agent': req.headers.get('User-Agent', ''),
        'content_type': req.headers.get('Content-Type', ''),
        'request_id': request_id,
        'body': req.get_data(as_text=True) if req.data else ''
    }
    
    return features

def analyze_request(features):
    """Run BERT model inference on request features ONLY (remove rules)"""
    bert_text = extract_request_text(features)
    logger.info(f"BERT text: {bert_text}")

    # Get BERT-based anomaly score
    bert_score = calculate_pll_score(bert_text)

    # Define threshold for malicious (e.g., 0.6)
    threshold = 0.6
    is_malicious = bert_score > threshold

    # For the response
    return {
        'is_malicious': is_malicious,
        'confidence': float(bert_score),
        'bert_score': float(bert_score),
        'threat_type': "anomaly_detected" if is_malicious else "benign",
        'detected_patterns': [],
        'bert_text': bert_text
    }


@app.route('/analyze', methods=['POST'])
def analyze_endpoint():
    """Main endpoint for request analysis"""
    try:
        # Extract features from the current request
        features = extract_features()
        
        # Run analysis
        analysis_result = analyze_request(features)
        
        # Create response
        response = {
            'timestamp': datetime.utcnow().isoformat(),
            'request_id': features['request_id'],
            'client_ip': features['client_ip'],
            'method': features['method'],
            'uri': features['uri'],
            **analysis_result
        }
        
        # Log the result
        if analysis_result['is_malicious']:
            logger.info(f"MALICIOUS REQUEST DETECTED: {json.dumps(response)}")
        else:
            logger.info(f"Benign request: {features['request_id']}")
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error analyzing request: {e}")
        return jsonify({
            'error': 'Analysis failed',
            'timestamp': datetime.utcnow().isoformat()
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    model_status = "loaded" if model is not None else "not_loaded"
    return jsonify({
        'status': 'healthy',
        'model_status': model_status,
        'device': DEVICE,
        'timestamp': datetime.utcnow().isoformat()
    })

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get model and system statistics"""
    return jsonify({
        'model_dir': MODEL_DIR,
        'device': DEVICE,
        'max_length': MAX_LEN,
        'num_masks': NUM_RANDOM_MASKS,
        'mlm_probability': MLM_PROB,
        'model_loaded': model is not None,
        'tokenizer_loaded': tokenizer is not None
    })

# Initialize the model when the app starts
if __name__ == '__main__':
    logger.info("Starting WAF Application with BERT integration")
    load_bert_model()
    app.run(host='0.0.0.0', port=5000, debug=False)
else:
    # For WSGI deployment
    load_bert_model()
