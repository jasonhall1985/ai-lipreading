const express = require('express');
const cors = require('cors');
const axios = require('axios');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');

const app = express();
const PORT = process.env.PORT || 3002;
const FLASK_URL = process.env.FLASK_URL || 'http://127.0.0.1:5004';

// Flask process
let flaskProcess = null;

// Middleware
app.use(cors());
app.use(express.json({ limit: '50mb' }));
app.use(express.static(path.join(__dirname, '../client/build')));

// Start Flask backend
const startFlaskBackend = () => {
  console.log('Starting Flask backend...');
  
  // Use the Python virtual environment we created
  const appPath = path.join(__dirname, '../app.py');
  const venvPythonPath = path.join(__dirname, '../venv/bin/python');
  
  console.log(`Starting Flask with app path: ${appPath}`);
  console.log(`Using Python from virtual environment: ${venvPythonPath}`);
  
  // Check if virtual environment exists
  if (fs.existsSync(venvPythonPath)) {
    flaskProcess = spawn(venvPythonPath, [appPath], {
      cwd: path.dirname(appPath)
    });
  } else {
    // Fall back to system Python if venv not found
    console.log('Virtual environment not found, falling back to system Python');
    try {
      flaskProcess = spawn('python3', [appPath], {
        shell: true,
        cwd: path.dirname(appPath)
      });
    } catch (error) {
      console.error(`Failed to start with python3: ${error.message}`);
      flaskProcess = spawn('python', [appPath], {
        shell: true,
        cwd: path.dirname(appPath)
      });
    }
  }

  flaskProcess.stdout.on('data', (data) => {
    console.log(`Flask stdout: ${data}`);
  });

  flaskProcess.stderr.on('data', (data) => {
    console.error(`Flask stderr: ${data}`);
  });

  flaskProcess.on('close', (code) => {
    console.log(`Flask backend exited with code ${code}`);
    // Restart Flask if it crashes
    if (code !== 0) {
      setTimeout(startFlaskBackend, 5000);
    }
  });
};

startFlaskBackend();

// API routes
app.get('/api/status', async (req, res) => {
  try {
    // Check Flask backend status
    const response = await axios.get(`${FLASK_URL}/health`, { timeout: 5000 });
    res.json({
      status: 'Server is running',
      flask_status: response.data.status,
      model_loaded: response.data.model_loaded
    });
  } catch (error) {
    console.error('Error checking Flask status:', error.message);
    res.json({
      status: 'Server is running',
      flask_status: 'not running',
      model_loaded: false
    });
  }
});

app.post('/api/lipread', async (req, res) => {
  try {
    const frame = req.body.frame;
    if (!frame) {
      return res.status(400).json({ error: 'No frame data provided' });
    }

    // Forward the frame to Flask
    const response = await axios.post(`${FLASK_URL}/predict`, { frame }, {
      timeout: 10000,
      headers: { 'Content-Type': 'application/json' }
    });
    
    res.json(response.data);
  } catch (error) {
    console.error('Error during lipreading:', error.message);
    
    // Check if Flask is not running
    if (error.code === 'ECONNREFUSED') {
      // Try to start Flask backend if it's not running
      if (!flaskProcess) {
        startFlaskBackend();
      }
      
      return res.status(503).json({
        error: 'Flask backend is not running',
        transcription: '',
        phrase: ''
      });
    }
    
    res.status(500).json({
      error: 'Failed to process lipreading',
      transcription: '',
      phrase: ''
    });
  }
});

app.post('/api/reset', async (req, res) => {
  try {
    // Forward the request to Flask
    const response = await axios.post(`${FLASK_URL}/reset`, {}, {
      timeout: 5000
    });
    
    res.json(response.data);
  } catch (error) {
    console.error('Error resetting frame buffer:', error.message);
    res.status(500).json({ error: 'Failed to reset frame buffer' });
  }
});

app.get('/api/phrases', async (req, res) => {
  try {
    // Forward the request to Flask
    const response = await axios.get(`${FLASK_URL}/phrases`, {
      timeout: 5000
    });
    
    res.json(response.data);
  } catch (error) {
    console.error('Error getting phrases:', error.message);
    res.status(500).json({ error: 'Failed to get phrases' });
  }
});

// Serve React app in production
if (process.env.NODE_ENV === 'production') {
  app.get('*', (req, res) => {
    res.sendFile(path.join(__dirname, '../client/build', 'index.html'));
  });
}

// Start server
app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
  
  // Start Flask backend
  startFlaskBackend();
});
