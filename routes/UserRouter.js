const express = require('express')
const router = express.Router()
const UserController = require('../controller/UserController')

router.post("/login", (req , res) =>{
    const login_user = {
        username : req.body.username,
        password : req.body.password
    }

    const Login = new UserController()

    const user_data = Login.login(login_user.username, login_user.password)

    if(user_data == null) return res.json({"message" : "error"}).status(401) 

    return res.json(user_data).status(200)
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

    const CheckUser = new UserController()

    res.json({"user_found" :  CheckUser.check_user(req.params.username)}).status(200)

})




module.exports = router