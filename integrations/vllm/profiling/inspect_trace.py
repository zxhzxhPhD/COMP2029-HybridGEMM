"""Small SQLite inventory for adapting the attribution parser to Nsight exports."""
import sqlite3
import sys

c = sqlite3.connect(sys.argv[1])
tables = [r[0] for r in c.execute("select name from sqlite_master where type='table'")]
for table in tables:
    if any(s in table for s in ['KERNEL', 'RUNTIME', 'NVTX', 'GRAPH', 'MEMCPY', 'MEMSET']):
        print(table, c.execute(f'select count(*) from {table}').fetchone()[0])
        print(c.execute(f'pragma table_info({table})').fetchall())
        print(c.execute(f'select * from {table} limit 1').fetchall())
print('Top kernels:')
print(c.execute('select s.value,count(*),sum(k.end-k.start)/1e6 from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.demangledName group by k.demangledName order by sum(k.end-k.start) desc limit 12').fetchall())
