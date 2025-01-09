#!/bin/sh

if [ ! -d /var/lib/openldap/openldap-data ]; then
    echo "Initializing LDAP database..."
    slapadd -F /etc/openldap/slapd.d -n 0 -l /etc/openldap/init.ldif
    chown -R ldap:ldap /var/lib/openldap/openldap-data /etc/openldap/slapd.d
fi

echo "Starting OpenLDAP server..."
exec slapd -F /etc/openldap/slapd.d -u ldap -g ldap -d 256
