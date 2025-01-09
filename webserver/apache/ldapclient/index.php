<?php
$ldapServer = "ldap://ldap-1";
$ldapPort = 389;
$ldapBaseDn = "ou=People,dc=plumbers,dc=it";
$adminDn = "cn=admin,dc=plumbers,dc=it";
$adminPassword = "admin";

$userCertificate = $_SERVER["SSL_CLIENT_CERT"] ?? null;
if (!$userCertificate) {
	die("No client certificate provided.");
}

$userCertificate = trim($userCertificate);
$userCertificateData = preg_replace('/-----.*-----/', '', $userCertificate);
$userCertificateData = base64_decode($userCertificateData);
$userCertificate = base64_encode($userCertificateData);

$ldapConn = ldap_connect($ldapServer, $ldapPort);
if (!$ldapConn) {
	die("Could not connect to LDAP server.");
}

ldap_set_option($ldapConn, LDAP_OPT_PROTOCOL_VERSION, 3);

if (!ldap_bind($ldapConn, $adminDn, $adminPassword)) {
	die("Admin bind failed: " . ldap_error($ldapConn));
}

$escaped = ldap_escape($userCertificate, "", LDAP_ESCAPE_FILTER);
$filter = "(userCertificate;binary=" . $escaped . ")";
$filter = "(objectClass=*)";
$attributes = ["dn", "cn", "userCertificate"];
$searchResult = ldap_search($ldapConn, $ldapBaseDn, $filter, $attributes);

if (!$searchResult) {
	die("Search failed: " . ldap_error($ldapConn));
}

$entries = ldap_get_entries($ldapConn, $searchResult);

$user = null;

foreach ($entries as $entry) {
	if ($userCertificate == base64_encode($entry["usercertificate;binary"][0])) {
		$user = $entry;
	}
}

if ($user == null) {
	die("Authentication Failed");
}

print("User Name: " . $user["cn"][0]);

ldap_unbind($ldapConn);
