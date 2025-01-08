FROM alpine:3.20.3

RUN apk add --no-cache \
		openldap \
		openldap-back-mdb \
		openldap-clients \
		;

RUN  install -m 755 -o ldap -g ldap -d /etc/openldap/slapd.d
