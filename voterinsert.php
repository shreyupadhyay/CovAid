<?php
$name=$_POST["name"];
$email=$_POST["email"];
$cont=$_POST["number"];
$password=$_POST["password"];
$confirmpass=$_POST["cpass"];
if($confirmpass!=$password)
{
	echo "<script type='text/javascript'>alert('Passwords dont match');</script>";
	echo "<script type='text/javascript'>window.open('votelogin.html');</script>";
}
else if(strlen($cont)!=10)
{
	echo "<script type='text/javascript'>alert('length of the phone number must be 10');</script>";
	echo "<script type='text/javascript'>window.open('votelogin.html');</script>";
}
else if (!filter_var($email, FILTER_VALIDATE_EMAIL))
{
	echo "<script type='text/javascript'>alert('wrong email format');</script>";
	echo "<script type='text/javascript'>window.open('votelogin.html');</script>";
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
else 
{
$sql="INSERT INTO votesignup(id,name,email,contactno,password) values('','$name','$email','$cont','$password')";

if (!mysqli_query($conn,$sql)) {

	echo("fail"); 
}
else
{
$to=$email;
$subject="Account verification";
$message="<a href='http://localhost/voting/verify.php'>Click here to verify your account</a>";
$headers="From: iwpvoting@gmail.com \r\n";
$headers .="MIME-Version: 1.0" . "\r\n";
$headers .="Content-type:text/html;charset=UTF-8" . "\r\n";
mail($to,"Account Verification",$message);
header("Location:thank.php");
}}
mysqli_close($conn);
}
?>