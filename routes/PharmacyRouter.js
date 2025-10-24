const express = require('express')
const router = express.Router()
const PharmacyController = require("../controller/PharmacyController")

router.post('/add_pharmacy', (req, res)=>{

    console.log(req.body)

    const data = {
        'name' : req.body.name,
        'address' : req.body.address,
        'email' : req.body.email,
        'openning_hours' : req.body.openning_hours,
        'contact_number' : req.body.contact_number,
        'latitude' : req.body.latitude,
        'longitude' : req.body.longitude
    }


    console.log(data)

    const Pharmacy = new PharmacyController()

    try {
        if(Pharmacy.pharmacy_add(data.name, data.address, data.email, data.openning_hours, data.contact_number, data.latitude, data.longitude)){
            res.json({message :  "added new pharmacy"})
        }
    } catch (error) {
        res.send(error)
    }
})

router.get("/get_pharmacies", async (req, res)=>{
    console.log("you are in the route to get all the pharmacies available in this server")

    const Pharmacy = new PharmacyController()

    try {
        const result =  await Pharmacy.pharmacy_all()
        console.log(result)
        res.json(result)

    } catch (error) {
        res.json(error)
    }
})



module.exports = router


