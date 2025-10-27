const express = require('express')
const router = express.Router()
const MedicineController = require('../controller/MedicineController')


router.get('/get_pharmacy_medicine/:id', async (req, res)=>{
    try {
        const pharma_id = req.params.id

        const med = new MedicineController()

        const items = await med.get_pharmacy_medicine(pharma_id)

        res.json(items).status(200)

    } catch (error) {
        console.log(error)
        res.send(error).status(500)
    }

})



module.exports = router