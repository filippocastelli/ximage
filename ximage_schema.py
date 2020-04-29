_XIMAGE_INDEX_CREATE_SCHEMA = """
-- Parse::SQL::Dia       version 0.27
-- Documentation         http://search.cpan.org/dist/Parse-Dia-SQL/
-- Environment           Perl 5.018002, /usr/bin/perl
-- Architecture          x86_64-linux-gnu-thread-multi
-- Target Database       sqlite3fk
-- Input file            ximage_schema.dia
-- Generated at          Tue Sep 19 16:21:30 2017
-- Typemap for sqlite3fk not found in input file

-- get_constraints_drop
drop index if exists idx_xccnc;
drop index if exists idx_ximxit;

-- get_permissions_drop

-- get_view_drop

-- get_schema_drop
drop table if exists XBlob;
drop table if exists XItem;
drop table if exists XImage;
drop table if exists XClass;
drop table if exists XImageParam;
drop table if exists XBelonging;

-- get_smallpackage_pre_sql

-- get_schema_create

create table XBlob (
   id            integer not null,
   xbelonging_id integer not null,
   parent_id     integer null    ,
   xclass_id     integer not null,
   val           real    not null,
   area          real    not null,
   vals          vector  not null,
   contour       points  not null,
   constraint pk_XBlob primary key (id),
   foreign key(xclass_id) references XClass(id) ,
   foreign key(parent_id) references XBlob(id) ,
   foreign key(xbelonging_id) references XBelonging(id)
)   ;

create table XItem (
   id uuid not null,
   constraint pk_XItem primary key (id)
)   ;

create table XImage (
   id   uuid not null,
   path text not null,
   constraint pk_XImage primary key (id)
)   ;

create table XClass (
   id      integer not null,
   classid integer not null,
   name    string  not null,
   color   color   not null,
   constraint pk_XClass primary key (id)
)   ;

create table XImageParam (
   ximage_id  uuid    not null,
   param_type integer not null,
   name       text    not null,
   val        xvalue  not null,
   constraint pk_XImageParam primary key (ximage_id,param_type,name),
   foreign key(ximage_id) references XImage(id)
)   ;

create table XBelonging (
   id        integer not null,
   ximage_id uuid    not null,
   xitem_id  uuid    not null,
   constraint pk_XBelonging primary key (id),
   foreign key(ximage_id) references XImage(id) ,
   foreign key(xitem_id) references XItem(id)
)   ;

-- get_view_create

-- get_permissions_create

-- get_inserts

-- get_smallpackage_post_sql

-- get_associations_create
create unique index idx_xccnc on XClass (classid,name,color) ;
create unique index idx_ximxit on XBelonging (ximage_id,xitem_id) ;
"""