# Spam Detection with DistilBERT

A machine learning project that uses DistilBERT to classify text messages as spam or ham (non-spam).

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
git clone https://github.com/yourusername/spam-detection.git
cd spam-detection
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your dataset file (CSV format) in the project directory
2. Update the config.yaml file with your settings
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
spam-detection/
├── app.py              # Main application file
├── model.py            # Model implementation
├── data.py             # Data loading and preprocessing
├── config.yaml         # Configuration file
├── requirements.txt    # Project dependencies
└── README.md          # This file
```

## License

MIT License

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request 