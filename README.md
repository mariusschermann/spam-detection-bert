# Spam Detection with DistilBERT

A powerful spam detection system using DistilBERT. Features include real-time classification, feedback system, and a user-friendly Gradio interface. Achieves 99.9% accuracy on test data.

## Features

- Uses DistilBERT for text classification
- Implements early stopping and learning rate scheduling
- Provides a Gradio web interface for easy interaction
- Includes feedback system for model improvement
- Supports mixed precision training
- Configurable through YAML file

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- Gradio
- Pandas
- NumPy
- scikit-learn

## Installation

1. Clone the repository:
```bash
git clone https://github.com/mariusschermann/spam-detection-bert.git
cd spam-detection-bert
```

2. Create and activate a virtual environment:
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset

The project uses the "SPAM text message 20170820 - Data.csv" dataset. The dataset should be placed in the root directory of the project. The CSV file should contain two columns:
- `Message`: The text message content
- `Category`: The label (spam/ham)

## Usage

1. Make sure your dataset file is in the project directory
2. (Optional) Update the config.yaml file with your settings
3. Run the application:
```bash
python app.py
```

4. Open your browser and navigate to the URL shown in the terminal (default: http://127.0.0.1:8080)

## Configuration

The project can be configured through `config.yaml`:

- Model settings (name, dropout, etc.)
- Training parameters (epochs, batch size, learning rate)
- Data split ratios
- Early stopping patience

## Project Structure

```
spam-detection-bert/
├── app.py              # Main application file
├── model.py            # Model implementation
├── data.py             # Data loading and preprocessing
├── feedback.py         # Feedback system implementation
├── config.yaml         # Configuration file
├── requirements.txt    # Project dependencies
└── README.md          # This file
```

## Troubleshooting

If you encounter any issues:

1. Make sure all dependencies are installed correctly
2. Check if the dataset file is in the correct location
3. Verify that your Python version is 3.8 or higher
4. Ensure you have enough disk space for the model files

## License

MIT License

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request 