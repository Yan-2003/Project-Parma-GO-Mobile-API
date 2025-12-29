const express = require('express')
const router = express.Router()
const DashboardController = require('../controller/DashboardController')

router.get('/get_all_med_count', async (req, res)=>{
    try {

        const Dashboard = new DashboardController()

        return res.json(await Dashboard.get_count_med()).status(200)

    } catch (error) {
        console.log(error)

        return res.json({"message" : error}).status(500)
    }
})

router.get('/get_all_pharma_count', async (req ,res)=>{
    try {
        const Dashboard = new DashboardController()

        return res.json(await Dashboard.get_all_pharma_count()).status(200)
    } catch (error) {
        console.log(error)
        return res.json({"message" : error}).status(500)
    }
})


module.exports = router