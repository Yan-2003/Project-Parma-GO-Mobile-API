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
            
            const query = await db.query(`
                SELECT 
                med.id as id,
                med.name as name,
                med.dosage_form as dosage_form,
                med.strength as strength,
                med.description as description,
                med.price as price,
                med.stock as price,
                med.expiration_date as expiration_date,
                pharma.name as pharma_name,
                pharma.latitude as latitude,
                pharma.longitude as longitude,
                pharma.address as address
                FROM medicine_tbl as med , pharmacy_tbl as pharma 
                WHERE LOWER(med.name) LIKE LOWER('${input}%') AND med.pharmacy_id = pharma.id
                                
            `) 
            console.log(query)
            console.log(query.rows)

            return query.rows

        } catch (error) {
            console.log(error)
        }


    }

    async get_med_by_id (id) {
        try {
            
            const query = await db.query(`
                SELECT * FROM medicine_tbl WHERE id = ${id}
            `)

            console.log(query.rows)
            return query.rows


        } catch (error) {
            console.log(error)
        }
    }

    async get_med_pharmacy (name) {
        try {
            
            const query = await db.query(`
                SELECT DISTINCT
                    pharma.id, 
                    pharma.name, 
                    pharma.longitude, 
                    pharma.latitude 
                FROM 
                    medicine_tbl as med,
                    pharmacy_tbl as pharma 
                WHERE 
                    med.name = '${name}' AND med.pharmacy_id = pharma.id 

            `)

            
            console.log(query.rows)

            return query.rows



        } catch (error) {
            console.log(error)
        }
    }


}


module.exports = Medicine