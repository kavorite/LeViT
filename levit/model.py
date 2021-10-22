from math import ceil

import tensorflow as tf
import tensorflow.keras as K
import tensorflow.keras.layers as L
from einops import rearrange


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def cast_tuple(val, l=3):
    val = val if isinstance(val, tuple) else (val,)
    return (*val, *((val[-1],) * max(l - len(val), 0)))


def always(val):
    def inner(*args, **kwargs):
        return val

    return inner


def serializable(cls):
    return tf.keras.utils.register_keras_serializable(
        package="kavorite/LeViT", name=cls.__name__
    )(cls)


@serializable
def hardswish(x):
    return 1 / 6 * x * tf.nn.relu6(x + 3)


@serializable
class SpatialPointwiseFFN(L.Layer):
    def __init__(
        self, depth, mult, dropout=0.0, name="pointwise_spatial_ffn", **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.expand = L.Conv2D(filters=depth * mult, kernel_size=1)
        self.contract = L.Conv2D(filters=depth, kernel_size=1)
        self.dropout = L.Dropout(dropout)

    def call(self, x):
        for op in [self.expand, self.dropout, self.contract, self.dropout]:
            x = op(x)
        return x

    def get_config(self):
        base = super().get_config()
        conf = dict(
            dim=self.contract.filters,
            mult=self.expand.filters // self.contract.filters,
            dropout=self.dropout.rate,
        )
        base.update(conf)
        return base


@serializable
class PatchConv(L.Layer):
    def __init__(self, depth, stride=1, norm_free=True, name="patch_conv", **kwargs):
        super().__init__(self, name=name, **kwargs)
        self.conv = L.Conv2D(
            filters=depth,
            kernel_size=1,
            strides=stride,
            use_bias=False,
        )
        self.norm = None if norm_free else L.BatchNormalization()

    def get_config(self):
        base = super().get_config()
        conf = dict(
            norm_free=self.norm is None,
            depth=self.conv.filters,
            stride=self.conv.strides,
        )
        base.update(conf)
        return base

    def call(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        return x


@serializable
class SpatialSelfAttention(L.Layer):
    def __init__(
        self,
        dim,
        fmap_size,
        heads=8,
        dim_key=32,
        dim_value=64,
        dropout=0.0,
        dim_out=None,
        downsample=False,
        norm_free=True,
        name="spatial_self_attention",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        inner_dim_key = dim_key * heads
        inner_dim_value = dim_value * heads
        dim_out = default(dim_out, dim)

        self.q_dim = ceil(fmap_size / (2 if downsample else 1))
        self.heads = heads
        self.scale = dim_key ** -0.5
        self.downsample = downsample
        self.conf = dict(
            dim=dim,
            fmap_size=fmap_size,
            heads=heads,
            dim_key=dim_key,
            dim_value=dim_value,
            dropout=dropout,
            dim_out=dim_out,
            downsample=downsample,
            norm_free=norm_free,
        )

        self.to_q = PatchConv(
            inner_dim_key,
            stride=2 if downsample else 1,
            name="embed_query",
            norm_free=norm_free,
        )
        self.to_k = PatchConv(inner_dim_key, name="embed_key", norm_free=norm_free)
        self.to_v = PatchConv(inner_dim_value, name="embed_value", norm_free=norm_free)
        self.attend = L.Activation(tf.nn.softmax, name="attend")
        self.to_out = K.Sequential(
            [
                L.Activation(hardswish),
                L.Conv2D(dim_out, kernel_size=1, padding="same"),
                L.Dropout(dropout),
            ]
        )
        if not norm_free:
            self.to_out.layers.insert(-1, L.BatchNormalization())
        self.pos_bias = L.Embedding(fmap_size ** 2, heads)
        q_range = tf.range(fmap_size, delta=(2 if downsample else 1))
        k_range = tf.range(fmap_size)
        q_pos = tf.stack(tf.meshgrid(q_range, q_range), axis=-1)
        k_pos = tf.stack(tf.meshgrid(k_range, k_range), axis=-1)

        q_pos, k_pos = map(lambda t: rearrange(t, "i j c -> (i j) c"), (q_pos, k_pos))
        rel_pos = tf.math.abs(q_pos[:, None, ...] - k_pos[None, :, ...])
        x_rel, y_rel = tf.unstack(rel_pos, axis=-1)
        self.pos_indices = tf.constant(x_rel * fmap_size + y_rel)

    def get_config(self):
        base = super().get_config()
        base.update(self.conf)
        return base

    def apply_pos_bias(self, fmap):
        bias = rearrange(self.pos_bias(self.pos_indices), "i j h -> h i j")
        return fmap + (bias / self.scale)

    def call(self, x):
        h = self.heads
        y = self.q_dim
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)
        q, k, v = (rearrange(t, "b ... (h d) -> b h (...) d", h=h) for t in (q, k, v))
        dots = tf.einsum("b h i d, b h j d -> b h i j", q, k)
        dots = self.apply_pos_bias(dots)
        attn = self.attend(dots)

        out = tf.einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b x y (h d)", h=h, y=y)
        return self.to_out(out)


@serializable
class SpatialTransformer(L.Layer):
    def __init__(
        self,
        dim,
        fmap_size,
        depth,
        heads,
        dim_key,
        dim_value,
        mlp_mult=2,
        dropout=0.0,
        dim_out=None,
        downsample=False,
        norm_free=False,
        name="spatial_transformer",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        dim_out = default(dim_out, dim)
        self.layers = []
        self.attn_residual = (not downsample) and dim == dim_out
        self.conf = dict(
            norm_free=norm_free,
            dim=dim,
            fmap_size=fmap_size,
            depth=depth,
            heads=heads,
            dim_key=dim_key,
            dim_value=dim_value,
            mlp_mult=mlp_mult,
            dropout=dropout,
            dim_out=dim_out,
            downsample=downsample,
            name=name,
        )

        for block in range(1, depth + 1):
            with tf.name_scope(name):
                self.layers.append(
                    [
                        SpatialSelfAttention(
                            dim,
                            fmap_size=fmap_size,
                            heads=heads,
                            dim_key=dim_key,
                            dim_value=dim_value,
                            dropout=dropout,
                            downsample=downsample,
                            dim_out=dim_out,
                            norm_free=norm_free,
                            name=f"block_{block}_self_attention",
                        ),
                        SpatialPointwiseFFN(
                            dim_out,
                            mlp_mult,
                            dropout=dropout,
                            name=f"block_{block}_pointwise_ffn",
                        ),
                    ]
                )

    def get_config(self):
        return self.conf

    def call(self, x):
        for attn, ff in self.layers:
            attn_res = x if self.attn_residual else 0
            x = attn(x) + attn_res
            x = ff(x) + x
        return x


def LeViT(
    *,
    image_size,
    dim,
    depth=4,
    heads=(4, 8, 6),
    mlp_mult=2,
    stages=3,
    dim_key=32,
    dim_value=64,
    dropout=0.0,
    norm_free=False,
    name=None,
    **kwargs,
):
    dims = cast_tuple(dim, stages)
    name = default(name, f"levit_{dims[0]}")
    depths = cast_tuple(depth, stages)
    layer_heads = cast_tuple(heads, stages)

    assert tf.reduce_all(
        [len(t) == stages for t in (dims, depths, layer_heads)]
    ), "dimensions, depths, and heads must be a tuple that is less than the designated number of stages"

    layers = [
        L.Conv2D(filters=32, kernel_size=3, strides=2, padding="valid"),
        L.ZeroPadding2D(),
        L.Conv2D(filters=64, kernel_size=3, strides=2, padding="valid"),
        L.ZeroPadding2D(),
        L.Conv2D(filters=128, kernel_size=3, strides=2, padding="valid"),
        L.ZeroPadding2D(),
        L.Conv2D(filters=dims[0], kernel_size=3, strides=2, padding="valid"),
    ]
    fmap_size = image_size // (2 ** 4)
    for idx, dim, depth, heads in zip(range(stages), dims, depths, layer_heads):
        is_last = idx == (stages - 1)
        stage = idx + 1
        layers.append(
            SpatialTransformer(
                dim,
                fmap_size,
                depth,
                heads,
                dim_key,
                dim_value,
                mlp_mult,
                dropout,
                norm_free=norm_free,
                name=f"stage_{stage}_self_attention",
            )
        )

        if not is_last:
            next_dim = dims[idx + 1]
            layers.append(
                SpatialTransformer(
                    dim,
                    fmap_size,
                    depth=1,
                    heads=heads * 2,
                    dim_key=dim_key,
                    dim_value=dim_value,
                    dim_out=next_dim,
                    downsample=True,
                    norm_free=norm_free,
                    name=f"stage_{stage}_shrink_attention",
                )
            )
            fmap_size = ceil(fmap_size / 2)

    levit = K.Sequential(layers=layers, name=name, **kwargs)
    levit(tf.random.normal([1, image_size, image_size, 3]))
    return levit
