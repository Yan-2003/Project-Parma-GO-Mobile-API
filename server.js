require('dotenv').config()
const express = require('express')
const { exec } = require('child_process')
const multer = require('multer')
const path = require('path')
const { v4: uuidv4 } = require('uuid')
const PORT = process.env.PORT

const app = express()

const storage = multer.diskStorage({
    destination: (req, file, cb) => cb(null, 'uploads/'),
    filename: (req, file, cb) => {
        const ext = path.extname(file.originalname)
        const uniqueName = uuidv4() + ext
        cb(null, uniqueName)
    }
})
const upload = multer({ storage })

app.post("/ocr", upload.single('image'), (req, res) => {
    if (!req.file) {
        return res.status(400).send("No image uploaded.")
    }

    const imagePath = path.resolve(req.file.path)
    exec(`python ./scripts/OCR_MODEL.py "${imagePath}"`, (error, stdout, stderr) => {
        if (error) {
            console.error("Exec error:", error)
            return res.status(500).send("Error: " + error.message)
        }
        if (stderr) {
            console.error("Python stderr:", stderr)
            return res.status(400).send("Stderr:" + stderr)
        }
        console.log(stdout)
        return res.status(200).send(stdout)
    })
})

app.listen(PORT, () => {
    console.log("Server is running at port: ", PORT)
})