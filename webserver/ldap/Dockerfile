FROM alpine:3.20.3

RUN apk add --no-cache \
    openldap \
    openldap-back-mdb \
    openldap-clients \
    ;

RUN install -m 755 -o ldap -g ldap -d /etc/openldap/slapd.d

COPY slapd.conf /etc/openldap/slapd.conf
COPY init.ldif /etc/openldap/slapd.d/init.ldif

EXPOSE 389

CMD ["slapd", "-h", "ldap://0.0.0.0:389", "-f", "/etc/openldap/slapd.conf", "-d", "256"]
