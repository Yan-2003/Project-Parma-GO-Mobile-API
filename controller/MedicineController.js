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


    async get_search_medicine(input){
        try{

            input = input.replace(/"/g, "")
            
            const query = await db.query(`SELECT * FROM medicine_tbl WHERE LOWER(name) LIKE LOWER('${input}%')`) 
            console.log(query)
            console.log(query.rows)

            return query.rows

        } catch (error) {
            console.log(error)
        }


    }


}


module.exports = Medicine