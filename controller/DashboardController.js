const db = require('../database/DB')


class DashboardController{

    async get_count_med () {

        try {
            const query = await db.query(
                `
                    SELECT COUNT(name) count_med FROM (SELECT DISTINCT name FROM medicine_tbl)
                `
            )
            
            return query.rows
        } catch (error) {
            console.log(error)
        }

    }


    async get_all_pharma_count () {

        try {
            const query = await db.query(
                `
                    SELECT COUNT(name) count_pharma FROM pharmacy_tbl
                `
            )


            return query.rows
        } catch (error) {
            console.log(error)
        }
    }

    async get_all_open_pharma_count () {
        try {
            const query = await db.query(
                `
                    SELECT COUNT(*) AS open_pharmacies
                    FROM pharmacy_tbl
                    WHERE CURRENT_TIME BETWEEN
                        to_timestamp(split_part(openning_hours, '-', 1), 'HH12:MIAM')::time
                    AND to_timestamp(split_part(openning_hours, '-', 2), 'HH12:MIAM')::time;
                `
            )

            return query.rows
        } catch (error) {
            console.log(error)
        }
    }
}




module.exports = DashboardController