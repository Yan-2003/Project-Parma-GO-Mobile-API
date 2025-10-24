require('dotenv').config()

const {Client} =  require('pg')


const DATABASE = new Client(
    {
        host: process.env.DB_HOST,
        user: process.env.DB_USERNAME,
        database : process.env.DB_DATABASE,
        password : process.env.DB_PASSWORD,
        port : process.env.DB_PORT
    }
)

if(DATABASE.connect()){
    console.log("✅ Database Connected.")
}else{
    console.log("Error while Connecting Database")
}




module.exports = DATABASE