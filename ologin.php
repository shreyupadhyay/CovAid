<?php
$user=$_POST['username'];
$pass=$_POST['pass'];
$host="localhost";
$dbusername="root";
$dbpassword="";
$dbname="opinions";
$conn=new mysqli($host,$dbusername,$dbpassword,$dbname);
$result=mysqli_query($conn,"SELECT * FROM org WHERE name='$user' AND password='$pass'");
$count=mysqli_num_rows($result);
if($count==1){ 
header('Location: homeorg.html');     }
		
else
{
		echo "<script type='text/javascript'>alert('Invalid name and password');</script>";
}
mysqli_close($conn);
?>
