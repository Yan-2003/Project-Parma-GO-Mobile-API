const db = require('../database/DB')

class Medicine{

    async get_pharmacy_medicine(pharmacy_id){
        try {
        const query =  await db.query(`SELECT * FROM medicine_tbl WHERE pharmacy_id =${pharmacy_id}`)
        console.log(query.rows)
        return query.rows
        } catch (error) {
            console.log(error)
        }
    }


}


module.exports = Medicine