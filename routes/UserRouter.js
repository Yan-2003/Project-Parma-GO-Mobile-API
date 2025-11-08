const express = require('express')
const router = express.Router()
const UserController = require('../controller/UserController')
const bcrypt = require('bcrypt')

router.post("/login", async (req , res) =>{
    const login_user = {
        username : req.body.username,
        password : req.body.password
    }

    console.log("user data: ", login_user)

    const Login = new UserController()

    const user_data = await Login.login(login_user.username, login_user.password)

    if(user_data){
        console.log("Authentication Successful")
        return res.json(user_data).status(200)
    }else{
        console.log("Authentication Failed")
        return res.json({mesasge : "login unsuccessful"}).status(401)
    }

})


router.post("/register", async (req, res)=>{

    const reg_user = {
        username : req.body.username,
        name : req.body.name,
        user_role : req.body.user_role,
        password : req.body.password
    }

    const user = new UserController()

    console.log(reg_user)

    if(await user.add_user(reg_user.username, reg_user.name , reg_user.user_role, reg_user.password)){
        return res.json({"message" : "successfully registered."}).status(200)
    }

    return res.json({"message" : "unable to register user"}).status(500)

})


router.get("/check_username/:username", async (req ,res)=>{

    console.log("checking username: ", req.params.username)

    const CheckUser = new UserController()

    console.log(await CheckUser.check_user(req.params.username))

    res.json({"user_found" : await CheckUser.check_user(req.params.username)}).status(200)

})




module.exports = router