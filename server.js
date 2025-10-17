require('dotenv').config();
const express = require('express');
const { exec } = require('child_process');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const { v4: uuidv4 } = require('uuid');

const PORT = process.env.PORT || 8000;
const app = express();

const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, 'uploads/'),
  filename: (req, file, cb) => {
    const ext = path.extname(file.originalname);
    const uniqueName = uuidv4() + ext;
    cb(null, uniqueName);
  },
});
const upload = multer({ storage });

app.post('/ocr', upload.single('image'), (req, res) => {
  if (!req.file) return res.status(400).json({ error: 'No image uploaded.' });

  const imagePath = path.resolve(req.file.path);

  exec(`python ./scripts/OCR_MODEL.py "${imagePath}"`, (error, stdout, stderr) => {
    // Remove temp uploaded file
    fs.unlinkSync(imagePath);

    if (error) {
      console.error('Exec error:', error);
      return res.status(500).json({ error: error.message });
    }

    try {
      const result = JSON.parse(stdout);
      return res.status(200).json(result);
    } catch (parseError) {
      console.error('Parse error:', parseError, 'stdout:', stdout);
      return res.status(500).json({ error: 'Failed to parse OCR output.' });
    }
  });
});

app.listen(PORT, () => {
  console.log(`Server is running at port: ${PORT}`);
});
