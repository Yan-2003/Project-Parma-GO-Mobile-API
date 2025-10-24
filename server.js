require('dotenv').config();
const express = require('express');
const { exec } = require('child_process');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const { v4: uuidv4 } = require('uuid');
const database_pg = require('./database/DB')

const PharmacyRouter = require("./routes/PharmacyRouter")


const PORT = process.env.PORT || 8000;
const app = express();


app.use(express.json());
app.use(express.urlencoded({ extended: true }));


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
  const pythonScript = path.resolve('./scripts/Pharma_YoloV8.py');


  const child = exec(`python "${pythonScript}" "${imagePath}"`, { timeout: 60000 }, (error, stdout, stderr) => {

    if (fs.existsSync(imagePath)) fs.unlinkSync(imagePath);

    if (error) {
      console.error('Exec error:', error);
      return res.status(500).json({ error: 'Error executing OCR script.' });
    }


    let jsonText = stdout.trim();
    const jsonMatch = jsonText.match(/\{[\s\S]*\}/); 
    if (!jsonMatch) {
      console.error('⚠️ No valid JSON found in Python output:', stdout);
      return res.status(500).json({ error: 'Invalid OCR output.' });
    }

    try {
      const result = JSON.parse(jsonMatch[0]);
      return res.status(200).json(result);
    } catch (parseError) {
      console.error('❌ JSON parse error:', parseError);
      console.error('RAW OUTPUT:', stdout);
      return res.status(500).json({ error: 'Failed to parse OCR output.' });
    }
  });

  child.stdout.on('data', (data) => console.log('PYTHON:', data.toString()));
  child.stderr.on('data', (data) => console.error('PYTHON ERR:', data.toString()));
});


// other routers

app.use('/pharmacy', PharmacyRouter)



app.listen(PORT, () => {
  console.log(`✅ Server is running at port: ${PORT}`);
});
