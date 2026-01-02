const db = require('../database/DB')


class Pharmacy{
    
    pharmacy_add(name, address, email, openning_hours, contact_number, latitude, longitude){

        db.query(`INSERT INTO pharmacy_tbl(name, address, email, contact_number, latitude, longitude, openning_hours) VALUES('${name}','${address}', '${email}', '${contact_number}', ${latitude}, ${longitude}, '${openning_hours}'  )`).then(e =>{
            console.log(e)
        }).catch(error=>console.log(error))
    }   
    

    async pharmacy_all (){
        try {
            const query = await db.query('SELECT * FROM pharmacy_tbl')
            console.log(query.rows)
            return query.rows
        } catch (error) {
            console.log(error)
            return error
        }

        
    }

    async get_pharamcy_by_id (id) {
        try {
            const query = await db.query(
                `
                    SELECT * FROM pharmacy_tbl WHERE id=${id}
                `
            )
            console.log(query.rows)
            return query.rows
        } catch (error) {
            console.log(error)
        }
    }





}

module.exports = Pharmacy;


