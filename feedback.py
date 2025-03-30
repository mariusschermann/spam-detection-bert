import json
import os
from datetime import datetime

class FeedbackSystem:
    def __init__(self, feedback_file="feedback.json"):
        self.feedback_file = feedback_file
        self.feedback_data = self.load_feedback()

    def load_feedback(self):
        """Load existing feedback data"""
        if os.path.exists(self.feedback_file):
            try:
                with open(self.feedback_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return {"flagged_messages": []}
        return {"flagged_messages": []}

    def save_feedback(self):
        """Save feedback data to file"""
        with open(self.feedback_file, 'w', encoding='utf-8') as f:
            json.dump(self.feedback_data, f, ensure_ascii=False, indent=2)

    def add_feedback(self, message, predicted_class, confidence, user_feedback):
        """Add new feedback entry"""
        feedback_entry = {
            "timestamp": datetime.now().isoformat(),
            "message": message,
            "predicted_class": "spam" if predicted_class == 1 else "ham",
            "confidence": confidence,
            "user_feedback": user_feedback
        }
        self.feedback_data["flagged_messages"].append(feedback_entry)
        self.save_feedback()

    def get_feedback_stats(self):
        """Get statistics about feedback"""
        total_flags = len(self.feedback_data["flagged_messages"])
        spam_flags = sum(1 for entry in self.feedback_data["flagged_messages"] 
                        if entry["predicted_class"] == "spam")
        ham_flags = total_flags - spam_flags
        
        return {
            "total_flags": total_flags,
            "spam_flags": spam_flags,
            "ham_flags": ham_flags
        } 