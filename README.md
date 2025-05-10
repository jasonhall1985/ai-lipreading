# AI Lipreading

This project implements a lipreading system using the AV-HuBERT model for audio-visual speech recognition.

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the AV-HuBERT model files:
```bash
curl -L https://dl.fbaipublicfiles.com/avhubert/model/ls960l/avhubert_base_ls960.pt -o avhubert_base_ls960.pt
curl -L https://dl.fbaipublicfiles.com/avhubert/model/ls960l/avhubert_base_ls960.yaml -o avhubert_base_ls960.yaml
```

## Usage

Run the demo server:
```bash
python server/lipnet/avhubert_demo.py
```

The server will start on port 8081. You can access the web interface at http://localhost:8081

## Project Structure

- `server/`: Server-side code
  - `lipnet/`: Lipreading model implementation
    - `avhubert_demo.py`: Demo server implementation
- `static/`: Static web assets
- `templates/`: HTML templates
- `config.json`: Configuration file

## Dependencies

- Python 3.9+
- PyTorch
- fairseq
- Flask
- OpenCV
- NumPy

## License

This project is licensed under the MIT License - see the LICENSE file for details. 