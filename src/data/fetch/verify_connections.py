# -*- coding: utf-8 -*-
"""验证 5 个数据源的连接与表结构(只读,不取数)。

用法:
    /home/intern_fjq_2026/miniconda3/envs/intern_fjq/bin/python \
        data_fetch/verify_connections.py
"""
import sys
import os
import socket

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg

import oracledb
import pymysql

PY = sys.executable
print(f"python : {PY}\n")


def report(ok, name, detail=""):
    print(f"[{'OK  ' if ok else 'FAIL'}] {name} {detail}")
    return ok


# ---------- 1) Oracle wind: 公司公告 + 调研问答 ----------
def check_wind():
    print("==== Oracle wind (10.23.153.15:21010/wind) ====")
    try:
        conn = oracledb.connect(user=cfg.WIND_USER, password=cfg.WIND_PWD, dsn=cfg.WIND_DSN)
    except Exception as e:
        report(False, "connect", f"-> {e}")
        return
    report(True, "connect", f"(oracledb {oracledb.__version__})")
    cur = conn.cursor()
    for key, t in cfg.WIND_TABLES.items():
        try:
            cur.execute(
                "SELECT column_name, data_type, data_length FROM all_tab_columns "
                "WHERE table_name = :t ORDER BY column_id", t=t)
            cols = cur.fetchall()
            if not cols:
                report(False, f"schema {key} ({t})", "no columns visible")
                continue
            report(True, f"schema {key} ({t})", f"{len(cols)} cols")
            for c in cols:
                print(f"    {c[0]:<28} {c[1]:<15} len={c[2]}")
        except Exception as e:
            report(False, f"schema {key} ({t})", f"-> {e}")
    conn.close()


# ---------- 3) MySQL datayes: 新闻三表 ----------
def check_mysql():
    print("\n==== MySQL datayes (10.80.139.20:3306) ====")
    try:
        conn = pymysql.connect(host=cfg.MYSQL_HOST, port=cfg.MYSQL_PORT,
                               user=cfg.MYSQL_USER, password=cfg.MYSQL_PWD,
                               database=cfg.MYSQL_DB, charset="utf8", connect_timeout=10)
    except Exception as e:
        report(False, "connect", f"-> {e}")
        return
    report(True, "connect", f"(pymysql {pymysql.__version__})")
    cur = conn.cursor()
    for key, t in cfg.MYSQL_TABLES.items():
        try:
            cur.execute(f"SHOW COLUMNS FROM `{t}`")
            cols = cur.fetchall()
            report(True, f"schema {key} ({t})", f"{len(cols)} cols")
            for c in cols:
                print(f"    {c[0]:<28} {c[1]}")
        except Exception as e:
            report(False, f"schema {key} ({t})", f"-> {e}")
    conn.close()


# ---------- 5) Oracle zyyx2: 研报 ----------
def check_zyyx2():
    print("\n==== Oracle zyyx2 (10.23.129.89:1521/zyyx2) ====")
    try:
        conn = oracledb.connect(user=cfg.ZYYX2_USER, password=cfg.ZYYX2_PWD, dsn=cfg.ZYYX2_DSN)
    except Exception as e:
        report(False, "connect", f"-> {e}")
        return
    report(True, "connect", f"(oracledb {oracledb.__version__})")
    cur = conn.cursor()
    t = cfg.ZYYX2_TABLE.split(".")[-1]
    try:
        cur.execute(
            "SELECT column_name, data_type, data_length FROM all_tab_columns "
            "WHERE table_name = :t ORDER BY column_id", t=t)
        cols = cur.fetchall()
        if not cols:
            report(False, f"schema ({cfg.ZYYX2_TABLE})", "no columns visible")
        else:
            report(True, f"schema ({cfg.ZYYX2_TABLE})", f"{len(cols)} cols")
            for c in cols:
                print(f"    {c[0]:<28} {c[1]:<15} len={c[2]}")
    except Exception as e:
        report(False, f"schema ({cfg.ZYYX2_TABLE})", f"-> {e}")
    conn.close()


# ---------- 1) FTP datayes: 社交文本 ----------
def check_ftp():
    print("\n==== FTP datayes (ftp.datayes.com:21) ====")
    host = cfg.FTP_HOST
    # 1) DNS
    try:
        ip = socket.gethostbyname(host)
        report(True, "dns", f"-> {ip}")
    except Exception as e:
        report(False, "dns", f"-> {e}")
    # 2) TCP 直连
    try:
        with socket.create_connection((host, cfg.FTP_PORT), timeout=8):
            report(True, "tcp direct", "port open")
    except Exception as e:
        report(False, "tcp direct", f"-> {e}")


if __name__ == "__main__":
    check_wind()
    check_mysql()
    check_zyyx2()
    check_ftp()
    print("\n==== 验证完成 ====")
