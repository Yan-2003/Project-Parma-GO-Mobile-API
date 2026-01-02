const express = require('express')
const router = express.Router()
const MedicineController = require('../controller/MedicineController')

const med = new MedicineController()

router.get('/get_pharmacy_medicine/:id', async (req, res)=>{
    try {
        const pharma_id = req.params.id

        const items = await med.get_pharmacy_medicine(pharma_id)

        res.json(items).status(200)

    } catch (error) {
        console.log(error)
        res.send(error).status(500)
    }

})

router.get('/get_pharmacy_meds/search', async (req, res)=>{
    try {
        
        const searchinput = req.query.input?.toLowerCase().trim() || " "

        console.log("Seach Input: " + searchinput)
        
        const items = await med.get_search_medicine(searchinput)

        res.json(items).status(200)

    } catch (error) {
        console.log(error)
    }
})

router.get('/get_medby_id/:id', async (req, res)=>{
    try {
        const med_by_id = await med.get_med_by_id(req.params.id)

        res.json(med_by_id).status(200)
    } catch (error) {
        console.log(error)
    }
})

router.get('/get_meds_pharma/:name', async ( req, res) =>{
    try {
        const get_pharmacy = await med.get_med_pharmacy(req.params.name)

        res.json(get_pharmacy).status(200)

    } catch (error) {   
        console.log(error)
    }
})

router.post('/add_medicine', async (req, res)=>{
    try {

        const data = req.body

        console.log(data)

        const add_med = await med.add_medicine(req.body)

        res.json({message : "POST added medicine", data  : add_med}).json(200)

    } catch (error) {
        console.log(error)
    }
})



module.exports = router