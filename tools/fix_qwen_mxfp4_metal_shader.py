#!/usr/bin/env python3
from pathlib import Path

p = Path("c/backend_metal.mm")
s = p.read_text()
marker = "fmt == 7) {                            // MXFP4 E2M1 + raw E8M0/32 columns"
if marker not in s:
    anchor = '''  } else if (fmt == 4) {                            // grouped int4: per-expert scale [O][ng]
    int rb=(K+1)/2, ng=(K+qgs-1)/qgs; device const uchar* w=(device const uchar*)(waddr[e])+(long)o*rb;
    device const float* sr=sc+(long)o*ng;           // grouped scales for this output row
    device const uchar4* w4=(device const uchar4*)w;
    for(int c=slane;c<K8;c+=32){ uchar4 b=w4[c];
      float4 w0=float4(float(int(b.x&0xF)-8),float(int(b.x>>4)-8),float(int(b.y&0xF)-8),float(int(b.y>>4)-8));
      float4 w1=float4(float(int(b.z&0xF)-8),float(int(b.z>>4)-8),float(int(b.w&0xF)-8),float(int(b.w>>4)-8));
      int g0=(8*c+0)/qgs,g1=(8*c+1)/qgs,g2=(8*c+2)/qgs,g3=(8*c+3)/qgs;
      int g4=(8*c+4)/qgs,g5=(8*c+5)/qgs,g6=(8*c+6)/qgs,g7=(8*c+7)/qgs;
      acc+=dot(w0*float4(sr[g0],sr[g1],sr[g2],sr[g3]),x4[2*c])
          +dot(w1*float4(sr[g4],sr[g5],sr[g6],sr[g7]),x4[2*c+1]); }
    for(int i=K8*8+slane;i<K;i+=32){ uchar b=w[i>>1]; int v=(i&1)?(b>>4):(b&0xF); acc+=float(v-8)*xr[i]*sr[i/qgs]; }
  } else { device const char* w=(device const char*)(waddr[e])+(long)o*K;
'''
    replacement = anchor[:-len('  } else { device const char* w=(device const char*)(waddr[e])+(long)o*K;\n')] + '''  } else if (fmt == 7) {                            // MXFP4 E2M1 + raw E8M0/32 columns
    int rb=(K+1)/2, ng=(K+31)/32;
    device const uchar* w=(device const uchar*)(waddr[e])+(long)o*rb;
    device const uchar* sr=(device const uchar*)(saddr[e])+(long)o*ng;
    const float mx4[16]={0.f,.5f,1.f,1.5f,2.f,3.f,4.f,6.f,-0.f,-.5f,-1.f,-1.5f,-2.f,-3.f,-4.f,-6.f};
    for(int i=slane*2;i<K;i+=64){
      uchar b=w[i>>1]; float sv=as_type<float>((uint)sr[i/32]<<23);
      acc += mx4[b&0xFu]*xr[i]*sv;
      if(i+1<K) acc += mx4[b>>4]*xr[i+1]*sv;
    }
  } else { device const char* w=(device const char*)(waddr[e])+(long)o*K;
'''
    if anchor not in s:
        raise SystemExit("batched fmt=4 anchor not found")
    s = s.replace(anchor, replacement, 1)
p.write_text(s)
