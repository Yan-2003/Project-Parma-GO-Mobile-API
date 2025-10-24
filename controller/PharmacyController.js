const db = require('../database/DB')


class Pharmacy{
    
    pharmacy_add(name, address, email, openning_hours, contact_number, latitude, longitude){

        db.query(`INSERT INTO pharmacy_tbl(name, address, email, contact_number, latitude, longitude, openning_hours) VALUES('${name}','${address}', '${email}', '${contact_number}', ${latitude}, ${longitude}, '${openning_hours}'  )`).then(e =>{
            console.log(e)
        })
    }   
    





}

module.exports = Pharmacy;


