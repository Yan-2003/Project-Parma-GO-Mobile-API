const express = require('express')
const router = express.Router()
const PharmacyController = require("../controller/PharmacyController")

router.post('/add_pharmacy', (req, res)=>{

    const data = {
        'name' : req.body.name,
        'address' : req.body.address,
        'email' : req.body.email,
        'openning_hours' : req.body.openning_hours,
        'contact_number' : req.body.contact_number,
        'latitude' : req.body.latitude,
        'longitude' : req.body.longitude
    }

    const Pharmacy = new PharmacyController()

    Pharmacy.pharmacy_add(data.name, data.address, data.email, data.openning_hours, data.contact_number, data.latitude, data.longitude)


    



})


