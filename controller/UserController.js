const db = require('../database/DB')
const bcrypt = require('bcrypt')


class User{


    async add_user(username, name, user_role, password){

        try {
            const salt = 10
    
            const hash_password = await bcrypt.hash(password, salt)

            const query = await db.query(`INSERT INTO user_tbl(username, name, password) VALUES('${username}', '${name}', '${hash_password}')`)
    
            console.log(query)
            return true
        
        } catch (error) {
            console.log(error)   
            return false 
        }
    }

    async check_user(username){

        try {
            const query = await db.query(`SELECT username FROM user_tbl WHERE username='${username}'`)

            console.log("result: ", query.rowCount)

            if(query.rowCount > 0){
                return true
            }

            return false
            
        } catch (error) {
            console.log(error)
            return false
        }

    }

    async login(username , password){

        
        try {
            const query = await db.query(`SELECT id, username, password, name, user_role FROM user_tbl WHERE username='${username}'`)

            if(query.rowCount == 0 ){
                 return false
            }

            console.log("username: " + username)
            console.log("password: ", password)

            console.log(query.rows[0])

            if(await bcrypt.compare(password, query.rows[0].password)) return query.rows[0]

        } catch (error) {

            console.log(error)

            return false
        }

    }

}

module.exports = User