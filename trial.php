<?php
$host="localhost";
$dbusername="root";
$dbpassword="";
$dbname="opinions";
$conn=mysqli_connect($host,$dbusername,$dbpassword,$dbname);
$query="CREATE TABLE votesignup(id int(100) NOT NULL AUTO_INCREMENT,name text(20) NOT NULL,email text(20),contactno INT(10) NOT NULL,password text(50) NOT NULL,PRIMARY KEY(id))";
if(!mysqli_query($conn,$query))
{
echo("error");
}
else
{
echo("success");
}
?>
