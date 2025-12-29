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

}

module.exports = DashboardController