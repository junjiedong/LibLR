CPPFLAGS=-std=c++14

IMPL_FILES=src/LogisticRegression.cc src/Utils.cc
LIB_FILES=src/CWrapper.cc $(IMPL_FILES)
TEST_FILES=src/main.cc $(IMPL_FILES)

RPATH_ARMA=-rpath /usr/local/lib/
RPATH_HDF5=-rpath /anaconda3/lib/
DLINK=-larmadillo

TARGETS=cpp_test liblr.so

lib:${LIB_FILES}
	g++ ${CPPFLAGS} $(RPATH_ARMA) $(RPATH_HDF5) $(DLINK) -fPIC -shared -o liblr.so $(LIB_FILES)

cpp_test:${TEST_FILES}
	g++ ${CPPFLAGS} $(RPATH_HDF5) $(DLINK) -o cpp_test $(TEST_FILES)

clean:
	rm -f $(TARGETS)
