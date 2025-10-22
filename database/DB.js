require('dotenv').config

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

DATABASE.connect()


module.exports = DATABASE