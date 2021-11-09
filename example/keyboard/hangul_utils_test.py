from hangul_utils import split_syllable_char, split_syllables, join_jamos

print(split_syllable_char(u"안"))

print(split_syllables(u"안"))


sentence = u"앞 집 팥죽은 붉은 팥 풋팥죽이고, 뒷집 콩죽은 햇콩 단콩 콩죽.우리 집 깨죽은 검은 깨 깨죽인데 사람들은 햇콩 단콩 콩죽 깨죽 죽먹기를 싫어하더라."
name = u"전다운"
s = split_syllables(sentence)
n = split_syllables(name)
print(s)
print(n)



sentence2 = join_jamos(s)
name2 = join_jamos(n)
print(sentence2)
print(name2)

li = ['ㄱ','ㅏ','ㅇ']

st = split_syllables(li)
print(st)
st2 = join_jamos(st)
print(st2)

print(sentence == sentence2)
print(name == name2)

