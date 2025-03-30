import gradio as gr
import torch
import yaml
import os
import logging
from transformers import AutoTokenizer
from model import SpamClassifier
from feedback import FeedbackSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set tokenizer parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class SpamDetectionApp:
    def __init__(self):
        """Initialize the application"""
        with open('config.yaml', 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device(self.config['training']['device'])
        self.model_name = self.config['model']['name']
        
        logger.info(f"Initializing with model: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = SpamClassifier(self.config)
        self.feedback_system = FeedbackSystem()
        self.model_loaded = False
        
        # Try to load existing model
        try:
            self.model.load_model()
            self.model_loaded = True
            logger.info("Model loaded successfully")
        except FileNotFoundError:
            logger.warning(f"No trained model found: {self.config['paths']['models']}/best_model.pt")
            logger.info("Starting training process...")
            self.model.train_model()
            self.model_loaded = True
            logger.info("Model trained and loaded successfully")
    
    def predict(self, text):
        """Make a prediction for the input text"""
        if not text:
            return "Please enter a message.", 0.0
        
        try:
            # Get prediction and confidence
            predicted_class, confidence = self.model.predict(text)
            
            # Convert to human-readable output
            result = "Spam" if predicted_class == 1 else "Ham"
            confidence_percent = confidence * 100
            
            return f"{result} (Confidence: {confidence_percent:.2f}%)", confidence_percent
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            return f"Error during prediction: {str(e)}", 0.0

    def handle_feedback(self, text, prediction, confidence, feedback):
        """Handle user feedback"""
        try:
            # Extract predicted class from prediction string
            predicted_class = 1 if "Spam" in prediction else 0
            
            # Add feedback
            self.feedback_system.add_feedback(
                message=text,
                predicted_class=predicted_class,
                confidence=confidence/100,  # Convert back to decimal
                user_feedback=feedback
            )
            
            # Get feedback statistics
            stats = self.feedback_system.get_feedback_stats()
            
            return f"Thank you for your feedback! Total flags: {stats['total_flags']}"
        except Exception as e:
            logger.error(f"Error handling feedback: {str(e)}")
            return f"Error saving feedback: {str(e)}"

def create_interface():
    """Create and return the Gradio interface"""
    app = SpamDetectionApp()
    
    # Create Gradio interface
    interface = gr.Interface(
        fn=app.predict,
        inputs=gr.Textbox(lines=3, placeholder="Enter your message here..."),
        outputs=[
            gr.Textbox(label="Result"),
            gr.Slider(minimum=0, maximum=100, label="Confidence (%)")
        ],
        title="Spam Detection",
        description="Enter a message to check if it's spam. Click 'Flag' if you think the prediction is incorrect.",
        examples=[
            ["WIN NOW! Limited time offer! Click here to claim your $1,000,000 prize!"],
            ["Hi, how are you? Let's meet tonight for dinner."],
            ["URGENT: Your bank account has been suspended. Click here to verify your identity."],
            ["Meeting at 2 PM in Conference Room 3."],
            ["FREE iPhone 15 Pro Max! Limited time offer! Sign up now!"]
        ],
        allow_flagging="manual"
    )
    
    return interface

if __name__ == "__main__":
    interface = create_interface()
    interface.launch(server_name="127.0.0.1", server_port=8080) 