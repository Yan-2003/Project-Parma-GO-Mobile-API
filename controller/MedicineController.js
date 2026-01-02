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
                    med.id,
                    med.name,
                    med.dosage_form,
                    med.strength,
                    med.description,
                    med.price,
                    med.stock,
                    med.expiration_date,
                    pharma.name AS pharma_name,
                    pharma.latitude,
                    pharma.longitude,
                    pharma.address,
                    similarity(LOWER(med.name), LOWER($1)) AS score
                FROM medicine_tbl med
                JOIN pharmacy_tbl pharma ON med.pharmacy_id = pharma.id
                WHERE 
                    -- fuzzy match for typos
                    similarity(LOWER(med.name), LOWER($1)) > 0.2

                    -- partial match for short queries like "a" or "ace"
                    OR LOWER(med.name) ILIKE '%' || LOWER($1) || '%'

                    -- optional: also search brand or descriptions
                    OR LOWER(med.brand) ILIKE '%' || LOWER($1) || '%'
                    OR LOWER(med.description) ILIKE '%' || LOWER($1) || '%'
                ORDER BY score DESC, med.name ASC
            `, [input]);
            
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

    async add_medicine (data) {
        try {

            console.log("Data to be inserted: ", data)



            const formatDate = (date) => {
                const yyyy = date.getFullYear()
                const mm = String(date.getMonth() + 1).padStart(2, '0')
                const dd = String(date.getDate()).padStart(2, '0')
                return `${yyyy}-${mm}-${dd}`
            }


            const query = await db.query(
                `
                    INSERT INTO medicine_tbl(name, description, brand , dosage_form, strength, price, stock, pharmacy_id, expiration_date, created_at)  
                    VALUES('${data.name}', '${data.description}', '${data.brand}', '${data.dosage_form}', '${data.strength}', ${data.price}, ${data.stock}, ${data.pharmacy_id}, '${formatDate(new Date(data.expiration_date))}', NOW())
                `
            )

            console.log(query)
            return query.command
        } catch (error) {
            console.log(error)
        }
    }


}


module.exports = Medicine