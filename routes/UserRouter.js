const express = require('express')
const router = express.Router()
const UserController = require('../controller/UserController')

router.post("/login", (req , res) =>{


})


router.post("/register", async (req, res)=>{
    const reg_user = {
        username : req.body.username,
        name : req.body.name,
        user_role : req.body.user_role,
        password : req.body.password
    }

    const user = new UserController()


    if(await user.add_user(reg_user.username, reg_user.name , reg_user.user_role, reg_user.password)){
        return res.json({"message" : "successfully registered."}).status(200)
    }

    return res.json({"message" : "unable to register user"}).status(500)

})


router.get("/check_username/:username", (req ,res)=>{

    //req.params.username

})




module.exports = router