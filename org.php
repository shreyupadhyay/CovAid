<?php
$name=$_POST['name'];
$cont=$_POST['cont'];
$address=$_POST['address'];
$password=$_POST["pass"];
$confirmpass=$_POST["cpass"];
if($confirmpass!=$password)
{
	echo "<script type='text/javascript'>alert('Passwords dont match');</script>";
	echo "<script type='text/javascript'>window.open('orglogin.html');</script>";
}
else if(strlen($cont)!=10)
{
	echo "<script type='text/javascript'>alert('length of the phone number must be 10');</script>";
	echo "<script type='text/javascript'>window.open('orglogin.html');</script>";
}
else
{
$host="localhost";
$dbusername="root";
$dbpassword="";
$dbname="opinions";
$conn=mysqli_connect($host,$dbusername,$dbpassword,$dbname);
if (!$conn) {
    die("Connection failed");
}
$query="INSERT INTO org(name,cont,address,password) VALUES ('$name','$cont','$address','$password')";

if(!mysqli_query($conn,$query))

{
	echo("failure");
}
else
{
	header("Location:thankorg.html");
}
mysqli_close($conn);
}
?>