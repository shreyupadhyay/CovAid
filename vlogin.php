<?php
$user=$_POST['username'];
$pass=$_POST['pass'];
$host="localhost";
$dbusername="root";
$dbpassword="";
$dbname="opinions";
$conn=new mysqli($host,$dbusername,$dbpassword,$dbname);
$result=mysqli_query($conn,"SELECT * FROM votesignup WHERE email='$user' AND password='$pass'");
$count=mysqli_num_rows($result);
if($count==1){ 
header('Location: home.html');
      }
		
else
{
		echo "<script type='text/javascript'>alert('Invalid email and password');</script>";
		echo"<script type='text/javascript'>window.location.replace('signupvoter.html')</script>";
}
mysqli_close($conn);
?>
